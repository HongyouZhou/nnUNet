"""Submit independent single-GPU tasks to either Charite A100 partition.

The site requires at least two GPUs per pgpu allocation. Grouped allocations
run independent one-GPU steps; they do not change model training to DDP.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


def excluded_nodes(inventory: str) -> list[str]:
    excluded = set()
    for line in inventory.splitlines():
        node, separator, resources = line.strip().partition("|")
        if not separator:
            continue
        types = re.findall(r"(?:^|,)gpu:([^:,()]+):\d+", resources)
        if not types or not all(value.startswith("nvidia_a100") for value in types):
            excluded.add(node)
    return sorted(excluded)


def script_options(script: Path) -> list[str]:
    options = []
    for line in script.read_text(encoding="utf-8").splitlines():
        if line.startswith("#SBATCH "):
            options.extend(shlex.split(line[len("#SBATCH ") :]))
    return options


def option_value(options: list[str], name: str, default: str = "") -> str:
    value = default
    for index, option in enumerate(options):
        if option.startswith(name + "="):
            value = option.split("=", 1)[1]
        elif option == name:
            value = options[index + 1]
    return value


def array_tasks(value: str) -> tuple[list[int], int]:
    specification, _, throttle = value.partition("%")
    tasks = []
    for entry in specification.split(","):
        if re.fullmatch(r"\d+-\d+", entry):
            first, last = map(int, entry.split("-"))
            tasks.extend(range(first, last + 1))
        elif entry.isdigit():
            tasks.append(int(entry))
        else:
            raise ValueError(f"Unsupported array specification: {value}")
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError(f"Invalid array specification: {value}")
    concurrency = min(len(tasks), int(throttle) if throttle else len(tasks))
    if concurrency < 1:
        raise ValueError("GPU task concurrency must be positive")
    return tasks, concurrency


def memory_megabytes(value: str) -> int:
    match = re.fullmatch(r"([1-9]\d*)([MG]?)", value)
    if match is None:
        raise ValueError(f"Expected memory in integer MB or GB, got {value!r}")
    return int(match[1]) * (1024 if match[2] == "G" else 1)


def candidates(script: Path, overrides: list[str], excluded: list[str]) -> dict[str, list[str]]:
    options = script_options(script) + overrides
    tasks, concurrency = array_tasks(option_value(options, "--array"))
    exclusion = set(excluded)
    exclusion.update(filter(None, option_value(options, "--exclude").split(",")))
    shared = ["--exclude=" + ",".join(sorted(exclusion))] if exclusion else []
    result = {
        "gpu": options + ["--partition=gpu", "--gres=gpu:1", *shared, str(script)],
    }
    # All allocations have the same size, with no unused GPUs and no increase
    # in the original array's maximum concurrent independent experiments.
    group_size = next(
        (size for size in (2, 3) if size <= concurrency and len(tasks) % size == 0),
        None,
    )
    if group_size is None:
        return result
    group_count = len(tasks) // group_size
    cpus = int(option_value(options, "--cpus-per-task", "1"))
    memory = memory_megabytes(option_value(options, "--mem"))
    exported = option_value(options, "--export", "ALL")
    if exported != "ALL" and not exported.startswith("ALL,"):
        raise ValueError("Grouped A100 submissions require --export=ALL")
    group_environment = (
        f"{exported},CORTICAL_GROUP_SCRIPT={script},"
        f"CORTICAL_GROUP_TASK_IDS={':'.join(map(str, tasks))},"
        f"CORTICAL_GROUP_SIZE={group_size},CORTICAL_GROUP_CPUS={cpus},"
        f"CORTICAL_GROUP_MEM_MB={memory}"
    )
    wrapper = script.parent / "run_separator_gpu_group.slurm"
    result["pgpu"] = options + [
        "--partition=pgpu",
        f"--gres=gpu:{group_size}",
        "--ntasks=1",
        f"--cpus-per-task={cpus * group_size}",
        f"--mem={memory * group_size}M",
        f"--array=0-{group_count - 1}%{max(1, concurrency // group_size)}",
        f"--export={group_environment}",
        *shared,
        str(wrapper),
    ]
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    dry_run = "--dry-run" in arguments
    if dry_run:
        arguments.remove("--dry-run")
    if not arguments or arguments[-1].startswith("-"):
        raise ValueError("Pass sbatch options followed by the GPU Slurm script")
    script = Path(arguments.pop()).resolve(strict=True)
    inventory = subprocess.check_output(
        ["sinfo", "-N", "-h", "-p", "gpu,pgpu", "-o", "%N|%G"], text=True
    )
    if not inventory.strip() or "nvidia_a100" not in inventory:
        raise RuntimeError("Unable to discover the Charite A100 inventory")
    choices = candidates(script, arguments, excluded_nodes(inventory))
    estimates = {}
    for partition, options in choices.items():
        probe = subprocess.run(
            ["sbatch", "--test-only", *options], capture_output=True, text=True
        )
        output = probe.stdout + probe.stderr
        if probe.returncode:
            print(f"[A100] {partition} unavailable: {output.strip()}", file=sys.stderr)
            continue
        match = re.search(r"to start at (\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)", output)
        estimates[partition] = match[1] if match else "9999-12-31T23:59:59"
    if not estimates:
        raise RuntimeError("Neither A100 allocation candidate passed sbatch --test-only")
    # Prefer the simpler individual allocation if Slurm gives equal or unknown
    # estimates. Scheduler estimates are advisory, not a start-time promise.
    selected = min(estimates, key=lambda key: (estimates[key], key != "gpu"))
    print(
        "[A100] " + json.dumps({"estimates": estimates, "selected": selected}),
        file=sys.stderr,
    )
    if dry_run:
        print(json.dumps({"estimates": estimates, "selected": selected, "commands": choices}))
        return 0
    return subprocess.run(["sbatch", *choices[selected]]).returncode


if __name__ == "__main__":
    raise SystemExit(main())
