#!/usr/bin/env python3
"""Run wflow for an AIFS project event + case via Julia.

Requires:
  - ~/.juliaup/bin/julia with @wflow1 project containing Wflow.jl
  - Model directory prepared by wflow_setup.py

Usage:
    python scripts/wflow_run.py dana_2024 hist_3h
    python scripts/wflow_run.py kristin_2026 ifs_jan22 --dry-run
"""

import sys
import os
import subprocess
import shutil
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

JULIA_CANDIDATES = [
    Path.home() / ".juliaup" / "bin" / "julia",
]


def detect_julia() -> Path | None:
    for candidate in JULIA_CANDIDATES:
        if candidate.exists():
            return candidate
    found = shutil.which("julia")
    return Path(found) if found else None


def strip_conda_ld_path(env: dict) -> dict:
    """Remove conda lib entries from LD_LIBRARY_PATH to avoid Julia/OpenSSL conflicts."""
    ld = env.get("LD_LIBRARY_PATH", "")
    filtered = ":".join(p for p in ld.split(":") if p and "conda" not in p)
    env["LD_LIBRARY_PATH"] = filtered
    return env


def main():
    parser = argparse.ArgumentParser(description="Run wflow for AIFS event")
    parser.add_argument("event_slug", help="Event slug, e.g. dana_2024")
    parser.add_argument("case",       help="Case name, e.g. hist_3h")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the Julia command without executing it",
    )
    args = parser.parse_args()

    slug = args.event_slug
    case = args.case

    model_dir = REPO_ROOT / "data" / "wflow" / slug / "model"
    toml_name = f"wflow_sbm_{slug}_{case}.toml"
    toml_path = model_dir / toml_name

    if not model_dir.exists():
        sys.exit(
            f"Model directory not found: {model_dir}\n"
            "Run wflow_setup.py first."
        )
    if not toml_path.exists():
        sys.exit(
            f"TOML not found: {toml_path}\n"
            "Run wflow_setup.py first."
        )

    julia = detect_julia()
    if julia is None:
        sys.exit(
            "Julia not found at ~/.juliaup/bin/julia or in PATH.\n"
            "Install juliaup or verify the PATH."
        )

    cmd = [
        str(julia),
        "--threads", "auto",
        "--project=@wflow1",
        "-e", "using Wflow; Wflow.run()",
        str(toml_path.resolve()),
    ]

    env = os.environ.copy()
    env = strip_conda_ld_path(env)
    env["HDF5_USE_FILE_LOCKING"] = "FALSE"

    print(f"Event     : {slug}")
    print(f"Case      : {case}")
    print(f"Model dir : {model_dir}")
    print(f"TOML      : {toml_path.name}")
    print(f"Julia     : {julia}")
    print(f"Command   : {' '.join(str(c) for c in cmd)}")
    print()

    if args.dry_run:
        print("[dry-run] Not executing.")
        return

    result = subprocess.run(cmd, cwd=str(model_dir), env=env)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
