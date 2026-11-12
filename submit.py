#!/usr/bin/env python3
"""
ELEC 402 Project 3 -- Submission helper.

    python3 submit.py start      # START
    python3 submit.py midpoint   # MIDPOINT, by 11:59 PM on December 1st
    python3 submit.py finish     # FINISH, by 11:59 PM on December 15th

Each one checks the required files, commits, and pushes its tag. See Submitting in the README.
"""

import subprocess
import sys
import argparse
from datetime import datetime, timezone
from pathlib import Path

COMMIT_PREFIX = "Project-3-ELEC402"
TOP, TAG = "mkMACBuff", "500"

# The design, the scripts and every result the graders rerun or read.
FLOW = {
    "submission/rtl":         [f"{TOP}.sv"],
    "submission/testbench":   ["rtl_checker.sv", "sdf_checker.sv"],
    "submission/scripts":     ["synth.tcl", "pnr.tcl"],
    "submission/syn_reports": [f"{TOP}_{TAG}_area.rpt", f"{TOP}_{TAG}_timing.rpt", f"{TOP}_{TAG}_qor.rpt",
                               f"{TOP}_{TAG}_power.rpt"],
    "submission/outputs":     [f"{TOP}_{TAG}_map.sv", f"{TOP}_{TAG}_map.sdf", f"{TOP}_{TAG}_map.sdc",
                               f"{TOP}_synth_{TAG}.dat"],
    "submission/pnr_reports": [f"{TOP}_{TAG}_drc.rpt", f"{TOP}_{TAG}_connectivity.rpt", f"{TOP}_{TAG}_antenna.rpt",
                               f"{TOP}_{TAG}_checkfiller.rpt", f"{TOP}_{TAG}_area.rpt", f"{TOP}_{TAG}_setup.rpt",
                               f"{TOP}_{TAG}_hold.rpt", f"{TOP}_{TAG}_power.rpt", f"{TOP}_{TAG}_skew_groups.rpt"],
    "submission/pnr_outputs": [f"{TOP}_{TAG}_pnr.v", f"{TOP}_{TAG}_pnr.sdf"],
    # every place-and-route stage's checkpoint (saveDesign writes a .enc file and a .enc.dat folder)
    "submission/pnr_outputs/checkpoints": [f"{TOP}_{TAG}_{st}.enc{ext}" for st in
                                           ("import", "floorplan", "power", "pins", "place", "cts", "route", "fill", "verify")
                                           for ext in ("", ".dat")],
}

# The full submission, required at MIDPOINT and at FINISH: the flow plus the report.
EVERYTHING = {
    **FLOW,
    "submission/report":      ["report.md"],
}

# phase -> (TAG name, required files or None, deadline text)
PHASES = {
    "start":    ("START",    None,       "when you start"),
    "midpoint": ("MIDPOINT", EVERYTHING, "11:59 PM on December 1st"),
    "finish":   ("FINISH",   EVERYTHING, "11:59 PM on December 15th"),
}

# The checkpoints in order: each one should already exist when the next is pushed
ORDER = ["START", "MIDPOINT", "FINISH"]

# Warn about any file this large: GitHub may refuse the push
TOO_BIG_MB = 50


def run(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def missing_files(required: dict) -> list[str]:
    missing = []
    for folder, patterns in required.items():
        for pattern in patterns:
            if not list(Path(folder).glob(pattern)):
                missing.append(f"{folder}/{pattern}")
    return missing


def warnings() -> list[str]:
    """Things that are allowed but are usually a mistake."""
    out = []
    rtl = Path("submission/rtl")
    extra = [f.name for f in rtl.glob("*") if f.is_file() and f.suffix not in (".sv", ".v", ".svh")
             and f.name != ".gitkeep" and not f.name.startswith(".")]
    if extra:
        out.append(f"submission/rtl/ has non-design files: {', '.join(sorted(extra))}")
    report = Path("submission/report/report.md")
    if report.exists():
        if report.stat().st_size < 500:
            out.append("submission/report/report.md is nearly empty")
        if not [f for f in Path("submission/report").glob("img/*") if not f.name.startswith(".")]:
            out.append("submission/report/img/ has no images (the report needs screenshots)")
    if Path("submission/report/report.pdf").exists() and not report.exists():
        out.append("found report.pdf: the report must be submission/report/report.md")
    for f in Path("submission").rglob("*"):
        if f.is_file() and f.stat().st_size > TOO_BIG_MB * 1024 * 1024:
            out.append(f"{f} is over {TOO_BIG_MB} MB and may not push")
    return out


def pushed_tags() -> set[str]:
    """Checkpoint tags already on GitHub (falls back to local tags when offline)."""
    r = run("git ls-remote --tags origin")
    if r.returncode == 0:
        return {line.split("refs/tags/")[1].removesuffix("^{}") for line in r.stdout.splitlines() if "refs/tags/" in line}
    return set(run("git tag -l").stdout.split())


def checkpoint_status(tag: str):
    print("\nTags:")
    have = pushed_tags()
    for t in ORDER:
        _, _, deadline = PHASES[t.lower()]
        mark = "pushed" if t in have else ("this one" if t == tag else "NOT pushed")
        print(f"  {t:9s} {mark:11s} deadline: {deadline}")


def do_phase(phase: str):
    tag, required, deadline = PHASES[phase]
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    msg = f"{COMMIT_PREFIX}: {tag}"
    print(f"{tag}  ({date})")
    print(f"Deadline: {deadline}.")

    checkpoint_status(tag)

    if phase == "midpoint":
        print(
            "\nMIDPOINT is a complete submission. If it passes everything (functional tests,\n"
            "synthesis timing, setup and hold after layout, DRC and connectivity clean, no\n"
            "filler gaps, both SDF checks), it can count as your final submission with full\n"
            "competitive marks. Otherwise it is free feedback and you continue to FINISH.\n"
        )

    if required:
        miss = missing_files(required)
        if miss:
            print("ERROR: missing required files:")
            for f in miss:
                print(f"  {f}")
            if any(not f.startswith("submission/report") for f in miss):
                print("Run the flow with bob.py first (see 'Running the flow with bob.py' in the README).")
            if any(f.startswith("submission/report") for f in miss):
                print("Write your report as submission/report/report.md (see Grading > Report in the README).")
            sys.exit(1)
        print("Files to be included:")
        for folder in required:
            for f in sorted(Path(folder).rglob("*")):
                if f.is_file() and f.name != ".gitkeep":
                    print(f"  {f}")

    warn = warnings()
    if warn:
        print("\nWarnings:")
        for w in warn:
            print(f"  {w}")

    if input(f"\nTag this commit as {tag} and push? [y/N] ").strip().lower() != "y":
        print("Aborted.")
        sys.exit(0)

    run("git add -A")

    result = run(f'git commit --allow-empty -m "{msg}"')
    if result.returncode != 0:
        print(f"ERROR: commit failed:\n{result.stderr}")
        sys.exit(1)

    result = run(f'git tag -a {tag} -m "{msg}" --force')
    if result.returncode != 0:
        print(f"ERROR: tagging failed:\n{result.stderr}")
        sys.exit(1)

    # push the branch, then (force-)move the tag to this commit
    result = run("git push origin HEAD")
    if result.returncode != 0:
        print(f"ERROR: push failed:\n{result.stderr}")
        print("Make sure you have push access to your repository.")
        sys.exit(1)

    result = run(f"git push origin {tag} --force")
    if result.returncode != 0:
        print(f"ERROR: tag push failed:\n{result.stderr}")
        sys.exit(1)

    print(f"done: commit '{msg}' tagged '{tag}' and pushed")


def main():
    parser = argparse.ArgumentParser(description="ELEC 402 Project 3 submission helper")
    sub = parser.add_subparsers(dest="command")
    for name in PHASES:
        sub.add_parser(name)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    do_phase(args.command)


if __name__ == "__main__":
    main()
