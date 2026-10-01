from pathlib import Path

from tools.charite_cortical.submit_a100 import candidates, excluded_nodes, option_value


def test_only_non_a100_nodes_are_excluded():
    inventory = "\n".join([
        "pcie40|gpu:nvidia_a100-pcie-40gb:1(S:0-7),tmp:2900G",
        "pcie80|gpu:nvidia_a100_80gb_pcie:1(S:4-5),tmp:1500G",
        "sxm40|gpu:nvidia_a100-sxm4-40gb:4(S:2-3),tmp:5800G",
        "sxm80|gpu:nvidia_a100-sxm4-80gb:8(S:4-7),tmp:28000G",
        "h100|gpu:nvidia_h100_80gb_hbm3:8(S:0-1),tmp:14000G",
        "l40s|gpu:nvidia_l40s:2(S:0-7),tmp:2900G",
    ])
    assert excluded_nodes(inventory) == ["h100", "l40s"]


def test_grouped_allocations_preserve_per_model_resources_and_concurrency():
    root = Path(__file__).resolve().parents[2]
    script = root / "slurm/charite_cortical/train_separator_continuity_pilot.slurm"
    choices = candidates(script, ["--parsable", "--mem=128G"], ["h100"])
    single, grouped = choices["gpu"], choices["pgpu"]
    assert option_value(single, "--gres") == "gpu:1"
    assert option_value(single, "--partition") == "gpu"
    assert option_value(grouped, "--gres") == "gpu:2"
    assert option_value(grouped, "--array") == "0-2%2"
    assert option_value(grouped, "--cpus-per-task") == "16"
    assert option_value(grouped, "--mem") == "262144M"
    assert "CORTICAL_GROUP_TASK_IDS=0:1:2:3:4:5" in option_value(grouped, "--export")
    assert "CORTICAL_GROUP_MEM_MB=131072" in option_value(grouped, "--export")


def test_smoke_groups_three_tasks_but_single_retry_never_reserves_idle_gpus():
    root = Path(__file__).resolve().parents[2]
    script = root / "slurm/charite_cortical/train_separator_continuity_smoke.slurm"
    grouped = candidates(script, [], ["h100"])["pgpu"]
    assert option_value(grouped, "--gres") == "gpu:3"
    assert option_value(grouped, "--array") == "0-0%1"
    assert option_value(grouped, "--cpus-per-task") == "48"
    assert option_value(grouped, "--mem") == "393216M"
    assert set(candidates(script, ["--array=2%1"], [])) == {"gpu"}
