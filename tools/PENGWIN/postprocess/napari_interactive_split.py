#!/usr/bin/env python3
"""Napari application for physician-guided, core-first instance splitting."""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import time
from pathlib import Path
from typing import Sequence

# This environment ships Qt5 without its Wayland plugin. XWayland plus GLX also
# avoids PyOpenGL losing the active context while vispy draws the first layer.
if sys.platform.startswith("linux") and os.environ.get("DISPLAY"):
    os.environ.setdefault("QT_QPA_PLATFORM", "xcb")
    os.environ.setdefault("PYOPENGL_PLATFORM", "glx")

import napari
import numpy as np
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QShortcut,
    QStyle,
    QVBoxLayout,
    QWidget,
)

from tools.PENGWIN.postprocess.cortical_anchored_split import SplitConfig
from tools.PENGWIN.postprocess.interactive_split_session import (
    InteractiveSplitSession,
    PromptKind,
    SplitPreview,
)


def _case_name(path: Path) -> str:
    name = path.name
    for suffix in (".nii.gz", ".nii", ".mha", ".mhd", ".nrrd"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return path.stem


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively split a 3D fragment instance segmentation in napari; "
            "ABBC is used only as hidden partition guidance."
        )
    )
    parser.add_argument("--ct", type=Path, required=True)
    parser.add_argument("--abbc", type=Path, required=True)
    parser.add_argument("--instances", type=Path, required=True)
    parser.add_argument("--gt", type=Path)
    parser.add_argument("--prob-label-3", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("interactive_split_output"))
    parser.add_argument("--case-name")
    parser.add_argument("--prompt-radius-mm", type=float, default=2.0)
    parser.add_argument("--core-erosion-mm", type=float, default=2.0)
    parser.add_argument("--core-anchor-min-voxels", type=int, default=20)
    parser.add_argument("--core-distinct-max-distance-mm", type=float, default=0.0)
    parser.add_argument("--min-split-piece-size", type=int, default=1_000)
    parser.add_argument("--max-graph-voxels", type=int, default=0)
    parser.add_argument("--bottleneck-band-mm", type=float, default=1.0)
    parser.add_argument(
        "--high-hu-cut-weight",
        type=float,
        default=0.0,
        help="Prefer cuts through high-HU non-cortical regions; zero disables it",
    )
    parser.add_argument(
        "--high-hu-center",
        type=float,
        default=500.0,
        help="HU value at the midpoint of the high-HU preference",
    )
    parser.add_argument(
        "--high-hu-scale",
        type=float,
        default=200.0,
        help="HU transition scale of the high-HU preference",
    )
    parser.add_argument(
        "--high-hu-protect-cortical",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Suppress the high-HU preference where ABBC predicts cortical bone",
    )
    parser.add_argument(
        "--coarse-to-fine",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--coarse-band-mm", type=float, default=12.0)
    parser.add_argument("--coarse-max-band-mm", type=float, default=24.0)
    parser.add_argument("--bottleneck-voxel-graph-max-nodes", type=int, default=100_000)
    parser.add_argument("--bottleneck-supervoxel-voxels", type=int, default=400)
    parser.add_argument("--bottleneck-max-graph-nodes", type=int, default=1_500_000)
    parser.add_argument("--initial-instance-id", type=int)
    parser.add_argument("--initial-source-zyx", type=int, nargs=3)
    parser.add_argument("--initial-sink-zyx", type=int, nargs=3)
    return parser


class InteractiveSplitWidget(QWidget):
    """Qt control surface connected to an InteractiveSplitSession."""

    def __init__(
        self,
        viewer: napari.Viewer,
        session: InteractiveSplitSession,
        output_dir: Path,
        case_name: str,
    ) -> None:
        super().__init__()
        self.viewer = viewer
        self.session = session
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.case_name = case_name
        self._syncing_points = False
        self._worker = None
        self._operation: str | None = None
        self._pending_split_radius: float | None = None

        affine = session.grid.affine_zyx()
        axis_labels = ("z", "y", "x")
        self.ct_layer = viewer.add_image(
            session.ct,
            name="CT",
            affine=affine,
            axis_labels=axis_labels,
            colormap="gray",
            contrast_limits=(-200.0, 1500.0),
        )
        if session.gt is not None:
            self.gt_layer = viewer.add_labels(
                session.gt,
                name="GT instances",
                affine=affine,
                axis_labels=axis_labels,
                opacity=0.35,
                visible=False,
            )
        else:
            self.gt_layer = None
        self.instances_layer = viewer.add_labels(
            session.instances,
            name="Instances (editable)",
            affine=affine,
            axis_labels=axis_labels,
            opacity=0.58,
        )
        self.instances_layer.selected_label = 0
        self.source_layer = viewer.add_points(
            np.empty((0, 3), dtype=float),
            ndim=3,
            name="Source prompt",
            affine=affine,
            axis_labels=axis_labels,
            size=4,
            face_color="#22c55e",
            border_color="#052e16",
            border_width=0.12,
            n_dimensional=True,
            out_of_slice_display=True,
        )
        self.sink_layer = viewer.add_points(
            np.empty((0, 3), dtype=float),
            ndim=3,
            name="Sink prompt",
            affine=affine,
            axis_labels=axis_labels,
            size=4,
            face_color="#e11d8a",
            border_color="#500724",
            border_width=0.12,
            n_dimensional=True,
            out_of_slice_display=True,
        )

        self._build_controls()
        self.instances_layer.events.selected_label.connect(self._on_selected_label)
        self.source_layer.events.data.connect(
            lambda event: self._on_points_changed("source")
        )
        self.sink_layer.events.data.connect(
            lambda event: self._on_points_changed("sink")
        )
        self._install_shortcuts()
        self._set_orientation("axial")
        self._center_on_segmentation()
        self._update_actions()

    def _build_controls(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        orientation_group = QGroupBox("View")
        orientation_layout = QHBoxLayout(orientation_group)
        orientation_layout.setContentsMargins(6, 6, 6, 6)
        self.orientation_buttons: dict[str, QPushButton] = {}
        button_group = QButtonGroup(self)
        button_group.setExclusive(True)
        for name, label in (
            ("axial", "Axial"),
            ("coronal", "Coronal"),
            ("sagittal", "Sagittal"),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked=False, orientation=name: self._set_orientation(orientation)
            )
            orientation_layout.addWidget(button)
            button_group.addButton(button)
            self.orientation_buttons[name] = button
        layout.addWidget(orientation_group)

        selection_group = QGroupBox("Fragment split")
        selection_layout = QGridLayout(selection_group)
        selection_layout.setContentsMargins(6, 6, 6, 6)
        self.select_button = QPushButton("Select instance")
        self.source_button = QPushButton("Place source")
        self.sink_button = QPushButton("Place sink")
        self.run_button = QPushButton("Run split")
        self.select_button.clicked.connect(self.activate_instance_selection)
        self.source_button.clicked.connect(lambda: self.activate_prompt("source"))
        self.sink_button.clicked.connect(lambda: self.activate_prompt("sink"))
        self.run_button.clicked.connect(self.run_split)
        self.run_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        selection_layout.addWidget(self.select_button, 0, 0, 1, 2)
        selection_layout.addWidget(self.source_button, 1, 0)
        selection_layout.addWidget(self.sink_button, 1, 1)
        selection_layout.addWidget(QLabel("Prompt radius"), 2, 0)
        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.0, 10.0)
        self.radius_spin.setSingleStep(0.5)
        self.radius_spin.setDecimals(1)
        self.radius_spin.setSuffix(" mm")
        self.radius_spin.setValue(2.0)
        selection_layout.addWidget(self.radius_spin, 2, 1)
        configured_high_hu_weight = float(
            self.session.cfg.bottleneck_high_hu_cut_weight
        )
        self.high_hu_checkbox = QCheckBox("Prefer high-HU cut")
        self.high_hu_checkbox.setChecked(configured_high_hu_weight > 0)
        self.high_hu_checkbox.setToolTip(
            "Lower the min-cut capacity in high-HU non-cortical regions"
        )
        selection_layout.addWidget(self.high_hu_checkbox, 3, 0, 1, 2)
        self.high_hu_weight_label = QLabel("High-HU weight")
        self.high_hu_weight_spin = QDoubleSpinBox()
        self.high_hu_weight_spin.setRange(0.05, 10.0)
        self.high_hu_weight_spin.setSingleStep(0.25)
        self.high_hu_weight_spin.setDecimals(2)
        self.high_hu_weight_spin.setValue(
            configured_high_hu_weight if configured_high_hu_weight > 0 else 1.0
        )
        self.high_hu_weight_spin.setEnabled(configured_high_hu_weight > 0)
        self.high_hu_weight_spin.setToolTip(
            "Larger values make high-HU non-cortical regions easier to cut"
        )
        selection_layout.addWidget(self.high_hu_weight_label, 4, 0)
        selection_layout.addWidget(self.high_hu_weight_spin, 4, 1)
        self.high_hu_checkbox.toggled.connect(self._set_high_hu_enabled)
        self.high_hu_weight_spin.valueChanged.connect(self._set_high_hu_weight)

        self.coarse_to_fine_checkbox = QCheckBox("Wide coarse search")
        self.coarse_to_fine_checkbox.setChecked(
            self.session.cfg.bottleneck_coarse_to_fine
        )
        self.coarse_to_fine_checkbox.setToolTip(
            "Search a wider region before voxel-level refinement"
        )
        self.coarse_to_fine_checkbox.toggled.connect(
            self._set_coarse_to_fine
        )
        selection_layout.addWidget(self.coarse_to_fine_checkbox, 5, 0, 1, 2)
        selection_layout.addWidget(self.run_button, 6, 0, 1, 2)
        layout.addWidget(selection_group)

        self.selection_label = QLabel("Instance: none")
        self.selection_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.selection_label)

        action_layout = QHBoxLayout()
        self.undo_button = QPushButton("Undo")
        self.redo_button = QPushButton("Redo")
        self.clear_button = QPushButton("Clear points")
        self.save_button = QPushButton("Save")
        self.undo_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowBack)
        )
        self.redo_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowForward)
        )
        self.save_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_DialogSaveButton)
        )
        self.undo_button.clicked.connect(self.undo_split)
        self.redo_button.clicked.connect(self.redo_split)
        self.clear_button.clicked.connect(self.clear_prompts)
        self.save_button.clicked.connect(self.save_results)
        action_layout.addWidget(self.undo_button)
        action_layout.addWidget(self.redo_button)
        action_layout.addWidget(self.clear_button)
        action_layout.addWidget(self.save_button)
        layout.addLayout(action_layout)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.progress.hide()
        layout.addWidget(self.progress)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.status_label)
        layout.addStretch(1)

    def _install_shortcuts(self) -> None:
        shortcuts = (
            ("I", self.activate_instance_selection),
            ("S", lambda: self.activate_prompt("source")),
            ("K", lambda: self.activate_prompt("sink")),
            ("Ctrl+Return", self.run_split),
            ("Ctrl+Z", self.undo_split),
            ("Ctrl+Y", self.redo_split),
            ("Ctrl+Shift+Z", self.redo_split),
            ("Ctrl+S", self.save_results),
        )
        self._shortcuts = []
        for sequence, callback in shortcuts:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def _center_on_segmentation(self) -> None:
        coordinates = np.nonzero(self.session.instances > 0)
        if coordinates[0].size:
            center = tuple(
                int((int(axis.min()) + int(axis.max())) // 2) for axis in coordinates
            )
            self.viewer.dims.current_step = center
        self.viewer.reset_view()
        self.viewer.camera.zoom *= 1.8

    def _set_orientation(self, orientation: str) -> None:
        orders = {
            "axial": (0, 1, 2),
            "coronal": (1, 0, 2),
            "sagittal": (2, 0, 1),
        }
        self.viewer.dims.ndisplay = 2
        self.viewer.dims.order = orders[orientation]
        self.orientation_buttons[orientation].setChecked(True)

    def _set_status(self, text: str, error: bool = False) -> None:
        color = "#b91c1c" if error else ""
        self.status_label.setStyleSheet(f"color: {color};" if color else "")
        self.status_label.setText(text)

    def _set_busy(self, busy: bool) -> None:
        self.progress.setVisible(busy)
        for button in (
            self.select_button,
            self.source_button,
            self.sink_button,
            self.run_button,
            self.undo_button,
            self.redo_button,
            self.clear_button,
            self.save_button,
        ):
            button.setEnabled(not busy)
        self.radius_spin.setEnabled(not busy)
        self.coarse_to_fine_checkbox.setEnabled(not busy)
        self.high_hu_checkbox.setEnabled(not busy)
        self.high_hu_weight_spin.setEnabled(
            not busy and self.high_hu_checkbox.isChecked()
        )
        if not busy:
            self._update_actions()

    def _set_coarse_to_fine(self, enabled: bool) -> None:
        self.session.cfg = dataclasses.replace(
            self.session.cfg,
            bottleneck_coarse_to_fine=bool(enabled),
        )

    def _set_high_hu_enabled(self, enabled: bool) -> None:
        self.high_hu_weight_spin.setEnabled(bool(enabled))
        self.session.cfg = dataclasses.replace(
            self.session.cfg,
            bottleneck_high_hu_cut_weight=(
                float(self.high_hu_weight_spin.value()) if enabled else 0.0
            ),
        )

    def _set_high_hu_weight(self, value: float) -> None:
        if not self.high_hu_checkbox.isChecked():
            return
        self.session.cfg = dataclasses.replace(
            self.session.cfg,
            bottleneck_high_hu_cut_weight=float(value),
        )

    def _update_actions(self) -> None:
        selected = self.session.selected_instance is not None
        self.source_button.setEnabled(selected)
        self.sink_button.setEnabled(selected)
        self.run_button.setEnabled(
            selected
            and self.session.source_point_zyx is not None
            and self.session.sink_point_zyx is not None
        )
        self.undo_button.setEnabled(bool(self.session.history))
        self.redo_button.setEnabled(bool(self.session.redo_stack))

    def _deactivate_edit_modes(self) -> None:
        self.instances_layer.mode = "pan_zoom"
        self.source_layer.mode = "pan_zoom"
        self.sink_layer.mode = "pan_zoom"

    def activate_instance_selection(self) -> None:
        if self._worker is not None:
            return
        self._deactivate_edit_modes()
        self.viewer.layers.selection.active = self.instances_layer
        self.instances_layer.selected_label = 0
        self.instances_layer.mode = "pick"
        self._set_status("Click an instance")

    def _on_selected_label(self, event=None) -> None:
        instance_id = int(self.instances_layer.selected_label)
        if instance_id <= 0:
            self.session.clear_selection()
            self._sync_prompt_layers()
            self.selection_label.setText("Instance: none")
            self._set_status("Background selected", error=True)
            self._update_actions()
            return
        try:
            self.session.select_instance_id(instance_id)
        except ValueError as error:
            self._set_status(str(error), error=True)
            return
        self._deactivate_edit_modes()
        self._sync_prompt_layers()
        voxel_count = int(np.count_nonzero(self.session.instances == instance_id))
        self.selection_label.setText(
            f"Instance: {instance_id} ({voxel_count:,} voxels)"
        )
        self._set_status(f"Selected instance {instance_id}")
        self._update_actions()

    def activate_prompt(self, kind: PromptKind) -> None:
        if self._worker is not None:
            return
        if self.session.selected_instance is None:
            self._set_status("Select an instance first", error=True)
            return
        self._deactivate_edit_modes()
        layer = self.source_layer if kind == "source" else self.sink_layer
        self.viewer.layers.selection.active = layer
        layer.mode = "add"
        self._set_status(f"Click once to place the {kind} point")

    def _point_layer(self, kind: PromptKind):
        return self.source_layer if kind == "source" else self.sink_layer

    def _session_point(self, kind: PromptKind):
        return (
            self.session.source_point_zyx
            if kind == "source"
            else self.session.sink_point_zyx
        )

    def _set_point_layer(self, kind: PromptKind, point) -> None:
        layer = self._point_layer(kind)
        self._syncing_points = True
        try:
            if point is None:
                layer.data = np.empty((0, 3), dtype=float)
            else:
                layer.data = np.asarray([point], dtype=float)
        finally:
            self._syncing_points = False

    def _sync_prompt_layers(self) -> None:
        self._set_point_layer("source", self.session.source_point_zyx)
        self._set_point_layer("sink", self.session.sink_point_zyx)

    def _on_points_changed(self, kind: PromptKind) -> None:
        if self._syncing_points:
            return
        layer = self._point_layer(kind)
        data = np.asarray(layer.data)
        if len(data) == 0:
            self.session.clear_prompt(kind)
            self._update_actions()
            return
        previous = self._session_point(kind)
        try:
            accepted = self.session.set_prompt(kind, data[-1])
        except ValueError as error:
            self._set_point_layer(kind, previous)
            self._set_status(str(error), error=True)
            self._update_actions()
            return
        self._set_point_layer(kind, accepted)
        layer.mode = "pan_zoom"
        self._set_status(f"{kind.capitalize()} point: {accepted}")
        self._update_actions()

    def clear_prompts(self) -> None:
        if self._worker is not None:
            return
        self.session.clear_prompts()
        self._sync_prompt_layers()
        self._set_status("Points cleared")
        self._update_actions()

    def initialize_interaction(
        self,
        instance_id: int,
        source_point_zyx: Sequence[int],
        sink_point_zyx: Sequence[int],
    ) -> None:
        """Preload a reproducible candidate interaction for visual review."""
        self.session.select_instance_id(instance_id)
        self.instances_layer.selected_label = int(instance_id)
        source = self.session.set_prompt("source", source_point_zyx)
        sink = self.session.set_prompt("sink", sink_point_zyx)
        self._sync_prompt_layers()
        voxel_count = int(np.count_nonzero(self.session.instances == instance_id))
        self.selection_label.setText(
            f"Instance: {int(instance_id)} ({voxel_count:,} voxels)"
        )
        midpoint = tuple(
            int(round((source[axis] + sink[axis]) / 2.0)) for axis in range(3)
        )
        self.viewer.dims.current_step = midpoint
        self._set_status("Candidate points loaded; run the split")
        self._update_actions()

    def run_split(self) -> None:
        if self._worker is not None:
            return
        try:
            radius = float(self.radius_spin.value())
            # Validate state before starting the worker.
            if self.session.selected_instance is None:
                raise ValueError("select an instance before running the split")
            if self.session.source_point_zyx is None or self.session.sink_point_zyx is None:
                raise ValueError("place both source and sink points")
        except ValueError as error:
            self._set_status(str(error), error=True)
            return

        self._deactivate_edit_modes()
        self._set_busy(True)
        self._set_status("Computing geodesic bottleneck cut...")
        self._operation = "split"
        self._pending_split_radius = radius
        print(
            "Split request: "
            f"instance={self.session.selected_instance}, "
            f"source={self.session.source_point_zyx}, "
            f"sink={self.session.sink_point_zyx}, radius_mm={radius}",
            flush=True,
        )
        worker = thread_worker(self.session.compute_split, ignore_errors=True)(radius)
        self._worker = worker
        worker.returned.connect(self._on_split_ready)
        worker.errored.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)
        worker.start()

    def _on_split_ready(self, preview: SplitPreview) -> None:
        try:
            event = self.session.commit_split(preview)
            self.instances_layer.data = self.session.instances
            self.instances_layer.selected_label = preview.selected_instance
            self._sync_prompt_layers()
            diagnostics = preview.diagnostics
            self.session.save_interactions(self._interaction_log_path())
            print(f"Split diagnostics: {diagnostics}", flush=True)
            method = diagnostics.get("peripheral_partition_method", "partition")
            self._set_status(
                f"Created instance {event['new_instance']} with {method}; "
                f"pieces: {diagnostics['source_voxels']:,} / "
                f"{diagnostics['sink_voxels']:,} voxels"
            )
        except Exception as error:  # Qt callbacks must not leak exceptions.
            self._set_status(f"Could not apply split: {error}", error=True)

    def _on_worker_error(self, error: Exception) -> None:
        if self._operation == "split" and self._pending_split_radius is not None:
            self.session.record_failed_split(error, self._pending_split_radius)
            self.session.save_interactions(self._interaction_log_path())
            print(f"Split failed: {error}", flush=True)
        self._set_status(f"Split failed: {error}", error=True)

    def _on_worker_finished(self) -> None:
        self._worker = None
        self._operation = None
        self._pending_split_radius = None
        self._set_busy(False)

    def undo_split(self) -> None:
        if self._worker is not None:
            return
        try:
            record = self.session.undo_last_split()
        except (ValueError, RuntimeError) as error:
            self._set_status(str(error), error=True)
            return
        self.instances_layer.refresh()
        self.instances_layer.selected_label = record.selected_instance
        self._sync_prompt_layers()
        self._set_status(f"Removed split instance {record.new_instance}")
        self._update_actions()

    def redo_split(self) -> None:
        if self._worker is not None:
            return
        try:
            record = self.session.redo_last_split()
        except (ValueError, RuntimeError) as error:
            self._set_status(str(error), error=True)
            return
        self.instances_layer.refresh()
        self.instances_layer.selected_label = record.selected_instance
        self._sync_prompt_layers()
        self._set_status(f"Restored split instance {record.new_instance}")
        self._update_actions()

    def save_results(self) -> None:
        if self._worker is not None:
            return
        output = self.output_dir / f"{self.case_name}_instances_edited.nii.gz"
        interaction_log = self._interaction_log_path()

        def save():
            return (
                self.session.save_instances(output),
                self.session.save_interactions(interaction_log),
            )

        self._set_busy(True)
        self._set_status("Saving...")
        self._operation = "save"
        worker = thread_worker(save, ignore_errors=True)()
        self._worker = worker
        worker.returned.connect(self._on_saved)
        worker.errored.connect(self._on_worker_error)
        worker.finished.connect(self._on_worker_finished)
        worker.start()

    def _on_saved(self, paths: tuple[Path, Path]) -> None:
        self._set_status(f"Saved {paths[0].name} and {paths[1].name}")

    def _interaction_log_path(self) -> Path:
        return self.output_dir / f"{self.case_name}_interactions.json"


def create_viewer(
    session: InteractiveSplitSession,
    output_dir: Path,
    case_name: str,
    prompt_radius_mm: float = 2.0,
    *,
    show: bool = True,
) -> tuple[napari.Viewer, InteractiveSplitWidget]:
    viewer = napari.Viewer(title=f"Interactive fragment split - {case_name}", show=show)
    widget = InteractiveSplitWidget(viewer, session, output_dir, case_name)
    widget.radius_spin.setValue(prompt_radius_mm)
    viewer.window.add_dock_widget(
        widget,
        name="Fragment split",
        area="right",
        tabify=False,
    )
    return viewer, widget


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.prompt_radius_mm < 0:
        raise ValueError("--prompt-radius-mm must be non-negative")
    if not np.isfinite(args.high_hu_cut_weight) or args.high_hu_cut_weight < 0:
        raise ValueError("--high-hu-cut-weight must be finite and non-negative")
    if not np.isfinite(args.high_hu_center):
        raise ValueError("--high-hu-center must be finite")
    if not np.isfinite(args.high_hu_scale) or args.high_hu_scale <= 0:
        raise ValueError("--high-hu-scale must be finite and positive")
    initial_values = (
        args.initial_instance_id,
        args.initial_source_zyx,
        args.initial_sink_zyx,
    )
    if any(value is not None for value in initial_values) and not all(
        value is not None for value in initial_values
    ):
        raise ValueError(
            "--initial-instance-id, --initial-source-zyx and --initial-sink-zyx "
            "must be provided together"
        )
    cfg = SplitConfig(
        core_anchor_erosion_mm=args.core_erosion_mm,
        core_anchor_min_voxels=args.core_anchor_min_voxels,
        core_anchor_distinct_max_distance_mm=args.core_distinct_max_distance_mm,
        min_split_piece_size=args.min_split_piece_size,
        core_first_max_graph_voxels=args.max_graph_voxels,
        bottleneck_band_width_mm=args.bottleneck_band_mm,
        bottleneck_high_hu_cut_weight=args.high_hu_cut_weight,
        bottleneck_high_hu_center_hu=args.high_hu_center,
        bottleneck_high_hu_scale_hu=args.high_hu_scale,
        bottleneck_high_hu_protect_cortical=args.high_hu_protect_cortical,
        bottleneck_coarse_to_fine=args.coarse_to_fine,
        bottleneck_coarse_band_width_mm=args.coarse_band_mm,
        bottleneck_coarse_max_band_width_mm=args.coarse_max_band_mm,
        bottleneck_voxel_graph_max_nodes=args.bottleneck_voxel_graph_max_nodes,
        bottleneck_supervoxel_target_voxels=args.bottleneck_supervoxel_voxels,
        bottleneck_max_graph_nodes=args.bottleneck_max_graph_nodes,
    )
    started = time.perf_counter()
    session = InteractiveSplitSession.from_files(
        args.ct,
        args.abbc,
        instances_path=args.instances,
        gt_path=args.gt,
        prob_label_3_path=args.prob_label_3,
        cfg=cfg,
    )
    elapsed = time.perf_counter() - started
    case_name = args.case_name or _case_name(args.abbc)
    print(
        f"Loaded {case_name}: shape={session.instances.shape}, "
        f"instances={int(session.instances.max())}, time={elapsed:.1f}s",
        flush=True,
    )
    _, widget = create_viewer(
        session,
        args.output_dir,
        case_name,
        prompt_radius_mm=args.prompt_radius_mm,
    )
    if args.initial_instance_id is not None:
        widget.initialize_interaction(
            args.initial_instance_id,
            args.initial_source_zyx,
            args.initial_sink_zyx,
        )
    napari.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
