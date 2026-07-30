from __future__ import annotations

from typing import Any

from .schema import CorticalContinuityHeadSchema


def build_cortical_continuity_network(
    configuration_manager: Any,
    num_input_channels: int,
    schema: CorticalContinuityHeadSchema,
    *,
    enable_deep_supervision: bool = False,
) -> Any:
    """Build the standard nnU-Net network body with a schema-sized flat head."""

    try:
        from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Building the cortical-continuity network requires a complete PyTorch nnU-Net environment"
        ) from exc

    if enable_deep_supervision:
        raise ValueError(
            "Cortical continuity v1 supervises C and directional A at full resolution; "
            "deep supervision is intentionally disabled"
        )
    return get_network_from_plans(
        configuration_manager.network_arch_class_name,
        configuration_manager.network_arch_init_kwargs,
        configuration_manager.network_arch_init_kwargs_req_import,
        num_input_channels,
        schema.total_channels,
        allow_init=True,
        deep_supervision=False,
    )
