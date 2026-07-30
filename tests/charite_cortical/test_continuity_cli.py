from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

from tools.charite_cortical.continuity_cli import main
from tools.charite_cortical.continuity_postprocess import (
    ContinuityInputs,
    save_inputs_npz,
)


class ContinuityCliTests(unittest.TestCase):
    def test_provenance_hashes_the_frozen_formal_artifact_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact_options = {
                "--source-manifest": root / "manifest.json",
                "--splits": root / "splits.json",
                "--continuity-config": root / "config.json",
                "--plans": root / "plans.json",
                "--cortical-checkpoint": root / "cortical.pth",
                "--provisional-checkpoint": root / "provisional.pth",
                "--abbc-config": root / "abbc.json",
            }
            for path in artifact_options.values():
                path.write_text("{}\n", encoding="utf-8")
            output = root / "run-metadata.json"
            arguments = ["provenance"]
            for option, path in artifact_options.items():
                arguments.extend((option, str(path)))
            arguments.extend(("--output", str(output)))
            with redirect_stdout(StringIO()):
                self.assertEqual(main(arguments), 0)
            value = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(value["provisional_model"]["fold"], "all")
            self.assertEqual(len(value["artifacts"]), 7)
            self.assertEqual(len(value["code"]["revision"]), 40)

    def test_formal_gate_requires_complete_runs_and_can_allow_neural_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def write_run(path: Path, mode: str) -> None:
                event_kinds = (
                    ("binary_1_2", "binary"),
                    ("multiway_1_2_3", "multiway"),
                    ("intact_1", "intact"),
                )
                results = []
                for event_id, kind in event_kinds:
                    results.append(
                        {
                            "event": {"event_id": event_id, "kind": kind},
                            "evaluation": {
                                "all_child_recovery": True,
                                "intact_false_split": False,
                            },
                            "cortical_grouping_evaluation": {
                                "prediction_count": 1 if kind == "intact" else (
                                    2 if kind == "binary" else 3
                                ),
                                "ground_truth_count": 1 if kind == "intact" else (
                                    2 if kind == "binary" else 3
                                ),
                                "all_child_recovery": True,
                            },
                        }
                    )
                path.write_text(
                    json.dumps(
                        {
                            "mode": mode,
                            "execute": True,
                            "input": {
                                "kind": "Dataset778_case",
                                "case_id": "1",
                                "direction_set": "axial19",
                            },
                            "manifest": {
                                "events": [
                                    {"event_id": event_id}
                                    for event_id, _ in event_kinds
                                ]
                            },
                            "results": results,
                        }
                    ),
                    encoding="utf-8",
                )

            o1 = root / "o1.json"
            o2 = root / "o2.json"
            write_run(o1, "O1")
            write_run(o2, "O2")
            output = root / "gate.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    main(
                        [
                            "gate",
                            "--o1",
                            str(o1),
                            "--o2-axial19",
                            str(o2),
                            "--expected-cases",
                            "1",
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            gate = json.loads(output.read_text(encoding="utf-8"))
            self.assertTrue(gate["neural_training_allowed"])
            self.assertEqual(gate["selected_o2_direction_set"], "axial19")

    def test_calibration_fails_closed_when_oof_evidence_is_insufficient(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calibration = root / "calibration.json"
            calibration.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "features": {
                                    "normalized_energy_gain": 1.0,
                                    "cluster_count": 2,
                                    "min_high_confidence_volume_mm3": 2.0,
                                    "min_repulsive_edges": 1,
                                    "min_piece_fraction": 0.25,
                                    "rim_support": 0.0,
                                    "unseeded_volume_components": 0,
                                },
                                "proposal_correct": True,
                                "intact_negative": False,
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            scorer = root / "scorer.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    main(
                        [
                            "calibrate",
                            "--input",
                            str(calibration),
                            "--output",
                            str(scorer),
                        ]
                    ),
                    0,
                )
            value = json.loads(scorer.read_text(encoding="utf-8"))
            self.assertEqual(value["type"], "always_abstain")
            self.assertEqual(value["reason"], "insufficient_calibration_samples")
            self.assertTrue((root / "scorer.fit.json").is_file())

    def test_refine_defaults_to_calibration_safe_abstention_and_evaluates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            provisional = np.ones((2, 3, 4), dtype=np.uint16)
            input_path = root / "input.npz"
            save_inputs_npz(
                input_path,
                ContinuityInputs(
                    provisional_instances=provisional,
                    cortex_probability=np.zeros(provisional.shape, dtype=np.float32),
                    affinity=np.zeros((1, *provisional.shape), dtype=np.float32),
                    affinity_offsets_zyx=np.asarray([[0, 0, 1]], dtype=np.int16),
                    spacing_zyx=np.asarray([0.5, 0.5, 0.5]),
                    affinity_kind="logits",
                ),
            )
            output = root / "refined.npz"
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "refine",
                            "--input",
                            str(input_path),
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            report = json.loads(stdout.getvalue())
            diagnostics = Path(report["diagnostics"])
            self.assertTrue(output.is_file())
            self.assertTrue(diagnostics.is_file())
            with np.load(output, allow_pickle=False) as payload:
                np.testing.assert_array_equal(
                    payload["full_instances"],
                    provisional,
                )
            diagnostic_value = json.loads(diagnostics.read_text(encoding="utf-8"))
            self.assertEqual(
                diagnostic_value["scorer"]["reason"],
                "calibrated_scorer_required",
            )

            ground_truth = root / "ground_truth.npy"
            np.save(ground_truth, provisional)
            evaluation_output = root / "evaluation.json"
            with redirect_stdout(StringIO()):
                self.assertEqual(
                    main(
                        [
                            "evaluate",
                            "--prediction",
                            str(output),
                            "--ground-truth",
                            str(ground_truth),
                            "--output",
                            str(evaluation_output),
                        ]
                    ),
                    0,
                )
            evaluation = json.loads(
                evaluation_output.read_text(encoding="utf-8")
            )
            self.assertEqual(
                evaluation["evaluations"]["0.5"]["panoptic_quality"],
                1.0,
            )
            self.assertIn("0.7", evaluation["evaluations"])

            stderr = StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(
                    main(
                        [
                            "refine",
                            "--input",
                            str(input_path),
                            "--output",
                            str(output),
                        ]
                    ),
                    2,
                )
            self.assertIn("refusing to overwrite", stderr.getvalue())

    def test_oracle_is_a_read_only_plan_without_execute(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fragments = np.zeros((1, 3, 6), dtype=np.uint16)
            fragments[:, :, :3] = 1
            fragments[:, :, 3:] = 2
            cortical = np.zeros_like(fragments)
            cortical[0, 1, 1] = 1
            cortical[0, 1, 4] = 2
            bundle = root / "oracle.npz"
            np.savez_compressed(
                bundle,
                full_fragment_instances=fragments,
                cortical_instances=cortical,
                spacing_zyx=np.asarray([1.0, 1.0, 1.0]),
            )
            output = root / "must-not-exist.json"
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(
                        [
                            "oracle",
                            "--input",
                            str(bundle),
                            "--mode",
                            "O1",
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            plan = json.loads(stdout.getvalue())
            self.assertFalse(plan["execute"])
            self.assertGreaterEqual(plan["event_counts"]["binary"], 1)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
