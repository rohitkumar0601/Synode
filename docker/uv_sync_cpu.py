"""`uv sync --frozen` a project into /opt/venv without the CUDA stack (CPU-only images).

The sdv and smartnoise locks resolve torch from PyPI, whose Linux wheel pulls ~4 GB of
nvidia-*/triton/cuda-* packages; xgboost pulls nvidia-nccl. Those are skipped
(--no-install-package) and, when torch came from PyPI, the CPU build of the SAME locked version
is installed from the PyTorch CPU index (--no-deps: torch's other deps are in the lock and
already installed). Every other package is exactly as locked. Locks that already pin torch
from the CPU index (synthcity) are left as they are.

Usage: python uv_sync_cpu.py <project-dir> [extra uv sync args...]
"""

from __future__ import annotations

import subprocess
import sys
import tomllib
from pathlib import Path

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CUDA_PREFIXES = ("nvidia-",)
CUDA_NAMES = {"triton", "cuda-bindings", "cuda-pathfinder", "cuda-toolkit"}


def main() -> None:
    project, extra = sys.argv[1], sys.argv[2:]
    lock = tomllib.loads((Path(project) / "uv.lock").read_text(encoding="utf-8"))
    packages = lock.get("package", [])
    skip = sorted({p["name"] for p in packages
                   if p["name"].startswith(CUDA_PREFIXES) or p["name"] in CUDA_NAMES})
    torch = [p for p in packages if p["name"] == "torch"
             and "pypi.org" in p.get("source", {}).get("registry", "")]
    if torch:
        skip.append("torch")
    cmd = ["uv", "sync", "--frozen", "--no-dev", "--no-editable", "--project", project, *extra]
    for name in skip:
        cmd += ["--no-install-package", name]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    if torch:
        version = torch[0]["version"]
        pip = ["uv", "pip", "install", "--python", "/opt/venv/bin/python", "--no-deps",
               "--index-url", CPU_INDEX, f"torch=={version}+cpu"]
        print("+", " ".join(pip), flush=True)
        subprocess.run(pip, check=True)


if __name__ == "__main__":
    main()
