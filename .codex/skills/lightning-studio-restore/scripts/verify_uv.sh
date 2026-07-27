#!/usr/bin/env bash
# Verify and optionally synchronize this repository's uv environment.
set -euo pipefail

sync=false
machine_check=false

usage() {
  cat <<'USAGE'
Usage: verify_uv.sh [--sync] [--machine-check]

--sync           Run uv sync before verification; use for a fresh clone.
--machine-check  Report CPU count, memory, and whether CUDA GPUs are available.
USAGE
}

while (($#)); do
  case "$1" in
    --sync) sync=true; shift ;;
    --machine-check|--gpu-check) machine_check=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

skill_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
repo_root=$(cd -- "$skill_dir/../../.." && pwd)
uv_dir="$repo_root/uv_env"
[[ -d "$uv_dir" && -f "$repo_root/notebooks/test_env.py" ]] || {
  printf 'Expected uv_env/ and notebooks/test_env.py under %s\n' "$repo_root" >&2; exit 1;
}

command -v uv >/dev/null || { printf 'uv is not available on PATH.\n' >&2; exit 1; }
cd "$uv_dir"

if "$sync"; then
  uv sync
fi

uv run python ../notebooks/test_env.py

if "$machine_check"; then
  uv run python - <<'PY'
import os
from pathlib import Path
import torch

print(f"cpu_logical_count={os.cpu_count()}")
meminfo = Path("/proc/meminfo")
if meminfo.exists():
    values = dict(
        line.split(":", 1) for line in meminfo.read_text().splitlines() if ":" in line
    )
    total_kib = int(values["MemTotal"].split()[0])
    available_kib = int(values["MemAvailable"].split()[0])
    print(f"memory_total_gib={total_kib / 1024 / 1024:.2f}")
    print(f"memory_available_gib={available_kib / 1024 / 1024:.2f}")
else:
    print("memory_total_gib=<unknown>")

cuda_available = torch.cuda.is_available()
print(f"cuda_available={cuda_available}")
if cuda_available:
    print(f"cuda_device_count={torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        print(f"cuda_device_{index}={properties.name}")
        print(f"cuda_device_{index}_memory_gib={properties.total_memory / 1024**3:.2f}")
else:
    print("cuda_device_count=0")
PY
fi
