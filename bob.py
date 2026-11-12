#!/usr/bin/env python3
"""
bob.py  --  ELEC 402 2026 Project 3

Run on ssh-soc.ece.ubc.ca. The CMC tool scripts are sourced automatically.

Step-by-step student workflow:

  Step 0 -- copy the script templates and fill in their TODOs:
    cp starter/scripts/synth.tcl submission/scripts/synth.tcl
    cp starter/scripts/pnr.tcl   submission/scripts/pnr.tcl

  Step 1 -- test your RTL functionally (runs your rtl_checker.sv):
    python3 bob.py check --run 500

  Step 2 -- synthesize the 500 MHz design:
    python3 bob.py synth --run 500

  Step 3 -- SDF timing check on the synthesized netlist:
    python3 bob.py timing --run 500 --stage syn

  Step 4 -- gate-level simulation to generate VCD, then power from the VCD:
    python3 bob.py sim --run 500
    python3 bob.py power --run 500

  Step 5 -- place and route (Innovus), all stages or one at a time:
    python3 bob.py pnr --run 500
    python3 bob.py pnr --run 500 --stage floorplan   (then power, pins, place, cts, route, fill, verify, finish)

  Step 6 -- SDF timing check on the post-layout netlist:
    python3 bob.py timing --run 500 --stage pnr

  Tidy up the files the tools leave behind (your design and results are kept):
    python3 bob.py clean          # add --all to also delete generated reports and outputs

Submission layout:
    submission/rtl/                mkMACBuff + your multiplier and adder tree
    submission/scripts/synth.tcl   synthesis script (from starter, TODOs filled in)
    submission/scripts/pnr.tcl     place and route script (from starter, TODOs filled in)
    submission/testbench/
        rtl_checker.sv  RTL functional checker (top: mkMACBuff_tb)
        sdf_checker.sv  gate-level SDF checker (top: mkMACBuff_sdf_checker)
"""

import glob
import json
import os
import re
import subprocess
import argparse
import sys
from pathlib import Path

TOP = "mkMACBuff"
TAGS = ["500"] # target frequencies in MHz; the tag names every output file
PNR_STAGES = ["import", "floorplan", "power", "pins", "place", "cts", "route", "fill", "verify", "finish"]

# Tool binaries (resolved from PATH after CMC env is loaded)
GENUS_CMD   = "genus"
INNOVUS_CMD = "innovus"
VSIM_CMD    = "vsim"
VLOG_CMD    = "vlog"

# Script paths relative to the project root
SYNTH_TCL = "submission/scripts/synth.tcl"      # student copy of starter/scripts/synth.tcl
PNR_TCL   = "submission/scripts/pnr.tcl"        # student copy of starter/scripts/pnr.tcl
POWER_TCL = "starter/scripts/saif2power.tcl"
VCD2SAIF  = "starter/scripts/vcd2saif_run.csh"

# Submission directories (students place their files here)
RTL_DIR  = "submission/rtl"
TB_DIR   = "submission/testbench"
STR_DIR  = "starter/rtl"
RPT_DIR  = "submission/syn_reports"
LOG_DIR  = "submission/syn_logs"
OUT_DIR  = "submission/outputs"
PNR_OUT  = "submission/pnr_outputs"
PNR_RPT  = "submission/pnr_reports"
PNR_LOG  = "submission/pnr_logs"

# Files the tools write into the project root; bob.py moves them into the logs folders after each run
GENUS_FILES   = ["genus.cmd*", "genus.log*", "fv"]
INNOVUS_FILES = ["innovus.cmd*", "innovus.log*", "innovus.logv*", "timingReports", "rc_model.bin",
                 "*.antenna.lef", "scheduling_file.cts*", "extLogDir", "*.ccopt*", "rcOut", "clock_tree_debug*"]

# GPDK45 gate-level cell models needed for post-synthesis and post-layout simulation
LIB_CELLS = (
    "/ubc/ece/data/cmc2/kits/GPDK45/gsclib045_all_v4.4"
    "/gsclib045/verilog/slow_vdd1v0_basicCells.v"
)

# Starter RTL compiled alongside student code (memory model, register array)
PROVIDED_RTL = [
    f"{STR_DIR}/memory_wrapper_2port.sv",
    f"{STR_DIR}/registerArray.sv",
]

# CMC tool environment scripts sourced before running Genus, Innovus or QuestaSim
CMC_SCRIPTS = [
    "/CMC/scripts/cadence.genus20.10.000.csh",
    "/CMC/scripts/siemens.questasim.2022.1_2.csh",
    "/CMC/scripts/cadence.innovus21.17.000.csh",
]

# Simulator messages that usually mean a real bug, not noise
SUSPECT_WARNINGS = ("vopt-2241", "Connection width does not match", "vlog-2623", "Undefined variable")


def _load_cmc_env():
    sources = "; ".join(f"source {s}" for s in CMC_SCRIPTS)
    cmd = f"{sources}; python3 -c 'import os,json; print(json.dumps(dict(os.environ)))'"
    r = subprocess.run(["tcsh", "-c", cmd],
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True)
    if r.returncode != 0:
        print(f"WARNING: could not source CMC tool scripts:\n{r.stderr[:400]}")
        return
    try:
        os.environ.update(json.loads(r.stdout))
    except Exception:
        print("WARNING: failed to parse environment from CMC scripts")


_LICENSE_ERRORS = (
    "Cannot find command filename",
    "license checkout failed",
    "Failed to check out",
    "FATAL: license",
    "Cannot checkout",
)


def run(cmd, desc="", check=True, capture=False):
    print(f"  {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    pipe = subprocess.PIPE if capture else None
    r = subprocess.run(cmd, shell=isinstance(cmd, str),
                       stdout=pipe, stderr=pipe, universal_newlines=True)
    if check and r.returncode != 0:
        print(f"ERROR: {desc or 'command'} failed (exit {r.returncode})")
        if capture:
            print(r.stdout[-3000:])
            print(r.stderr[-1000:])
        sys.exit(1)
    return r


def check_tool(name):
    r = subprocess.run(["which", name], stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE, universal_newlines=True)
    if r.returncode != 0:
        print(f"ERROR: '{name}' not found in PATH -- source the CMC scripts first")
        sys.exit(1)


def rtl_files():
    """Every .sv in submission/rtl/. Stops if a module is defined twice (a leftover copy would
    silently replace your real design, depending on file order)."""
    # skip macOS metadata files (._name.sv) that tar and some copy tools create
    files = sorted(f for f in Path(RTL_DIR).glob("*.sv") if not f.name.startswith("._"))
    if not (Path(RTL_DIR) / f"{TOP}.sv").exists():
        print(f"ERROR: {RTL_DIR}/{TOP}.sv not found -- place your implementation there")
        sys.exit(1)
    seen = {}
    for f in files:
        for m in re.finditer(r"^\s*module\s+(\w+)", f.read_text(errors="replace"), re.M):
            if m.group(1) in seen:
                print(f"ERROR: module {m.group(1)} is defined in both {seen[m.group(1)]} and {f}.")
                print(f"       Keep only your design files in {RTL_DIR}/ (testbenches go in {TB_DIR}/).")
                sys.exit(1)
            seen[m.group(1)] = f
    return [str(f) for f in files]


def sim_netlist(path):
    """Gate-level netlists lose the N parameter, so the testbenches' #(.N(N)) would not load. Write a
    simulation copy (<name>_sim.sv) with N added back and return its path. The real netlist is left
    unchanged, because Innovus does not accept the parameter in its netlist reader. N's value does not
    matter here: the netlist widths are already fixed."""
    path = Path(path)
    text = path.read_text(errors="replace")
    fixed = re.sub(rf"\bmodule {TOP}_\d+\b", f"module {TOP}", text)
    if not re.search(rf"\bmodule {TOP}\s*#", fixed):
        fixed = re.sub(rf"\bmodule {TOP}\s*\(", f"module {TOP} #(parameter N = 1) (", fixed, count=1)
    out = path.with_name(path.stem + "_sim.sv")
    out.write_text(fixed)
    return out


def sweep(dest, patterns):
    """Move the files a tool left in the project root into dest (replacing older copies)."""
    import shutil
    Path(dest).mkdir(parents=True, exist_ok=True)
    for p in sorted({f for pat in patterns for f in glob.glob(pat)}):
        target = Path(dest) / Path(p).name
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
        shutil.move(p, str(target))


def clock_latency(tag):
    """Average clock insertion delay (ns, slow corner) from the CTS skew report, or None."""
    rpt = Path(PNR_RPT) / f"{TOP}_{tag}_skew_groups.rpt"
    if not rpt.exists():
        print(f"  WARNING: {rpt} not found -- post-layout check runs without clock latency")
        return None
    for line in rpt.read_text(errors="replace").splitlines():
        f = line.split()
        if f and f[0].startswith("slow") and "setup.late" in f[0] and len(f) > 5:
            try:
                return float(f[5])     # Avg ID column
            except ValueError:
                pass
    return None


def warn_suspect(text):
    hits = [l.strip() for l in text.splitlines() if any(w in l for w in SUSPECT_WARNINGS)]
    if hits:
        print("\n  WARNING: the simulator reported messages that usually mean a real bug:")
        for h in hits[:8]:
            print(f"    {h}")
        print("  (for example an undeclared signal becomes a 1-bit wire)\n")


def cmd_synth(args):
    tag = args.run
    print(f"=== Synthesis: {tag} MHz ===")
    check_tool(GENUS_CMD)
    rtl_files()
    if not Path(SYNTH_TCL).exists():
        print(f"ERROR: {SYNTH_TCL} not found -- copy starter/scripts/synth.tcl there "
              f"and fill in its TODOs")
        sys.exit(1)
    for d in (RPT_DIR, LOG_DIR, OUT_DIR):
        Path(d).mkdir(parents=True, exist_ok=True)

    preamble = f"set RUN_TAG {tag}; "
    log_path = Path(LOG_DIR) / f"genus_{tag}.log"
    print(f"  logging to {log_path}")
    r = subprocess.run(
        [GENUS_CMD, "-batch", "-execute", f"{preamble}source {SYNTH_TCL}"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    log_path.write_text(r.stdout)
    sweep(LOG_DIR, GENUS_FILES)
    print(r.stdout[-5000:] if len(r.stdout) > 5000 else r.stdout)
    if r.returncode != 0:
        print(f"ERROR: genus synthesis failed (exit {r.returncode})")
        sys.exit(1)

    print(f"reports -> {RPT_DIR}/")
    print(f"logs    -> {LOG_DIR}/")
    print(f"outputs -> {OUT_DIR}/")


def cmd_check(args):
    print(f"=== RTL check ({args.run} MHz) ===")
    check_tool(VLOG_CMD)
    check_tool(VSIM_CMD)

    sv_files = rtl_files()
    tb_file = str(Path(TB_DIR) / "rtl_checker.sv")
    if not Path(tb_file).exists():
        print(f"ERROR: {tb_file} not found -- copy starter/tb/rtl_checker.sv there")
        sys.exit(1)

    # RTL sim: memory wrappers + student RTL + testbench (no lib cells, no netlist)
    r = run([VLOG_CMD, "-sv"] + PROVIDED_RTL + sv_files + [tb_file],
            desc="vlog", check=False, capture=True)
    if r.returncode != 0:
        print("  COMPILE FAIL")
        print(r.stdout[-2000:])
        sys.exit(1)
    warn_suspect(r.stdout)

    r = run([VSIM_CMD, "-batch", "-do", "run -all; quit -f", f"work.{TOP}_tb"],
            check=False, capture=True)
    out = r.stdout + r.stderr
    print(out[-4000:])
    warn_suspect(out)
    print(f"=== Overall: {'PASS' if 'ALL PASS' in out else 'FAIL'} ===")


def cmd_sim(args):
    tag = args.run
    print(f"=== Gate-level sim for power ({tag} MHz) ===")
    check_tool(VLOG_CMD)
    check_tool(VSIM_CMD)

    tb_file = str(Path(TB_DIR) / "rtl_checker.sv")
    netlist = Path(OUT_DIR) / f"{TOP}_{tag}_map.sv"
    vcd_file = Path(OUT_DIR) / f"{TOP}_{tag}.vcd"
    if not Path(tb_file).exists():
        print(f"ERROR: {tb_file} not found")
        sys.exit(1)
    if not netlist.exists():
        print(f"ERROR: {netlist} not found -- run synth first")
        sys.exit(1)

    netlist = sim_netlist(netlist)
    # The netlist already contains the full design, so the RTL is not compiled.
    run([VLOG_CMD, "-sv", LIB_CELLS] + PROVIDED_RTL + [str(netlist), tb_file],
        desc="vlog compile")
    do_cmd = (f"vcd file {vcd_file}; vcd add /{TOP}_tb/DUT/*; "
              f"run -all; vcd flush; quit -f")
    run([VSIM_CMD, "-batch", "-voptargs=+acc", "-do", do_cmd, f"work.{TOP}_tb"],
        desc="vsim gate-level")
    print(f"VCD -> {vcd_file}")


def cmd_timing(args):
    stage, tag = args.stage, args.run
    print(f"=== SDF timing ({'synthesized' if stage == 'syn' else 'post-layout'} netlist) ===")
    check_tool(VLOG_CMD)
    check_tool(VSIM_CMD)

    if stage == "syn":
        netlist = Path(OUT_DIR) / f"{TOP}_{tag}_map.sv"
        sdf_file = Path(OUT_DIR) / f"{TOP}_{tag}_map.sdf"
        hint = "run synth first"
    else:
        netlist = Path(PNR_OUT) / f"{TOP}_{tag}_pnr.v"
        sdf_file = Path(PNR_OUT) / f"{TOP}_{tag}_pnr.sdf"
        hint = "run pnr first"
    for f in (netlist, sdf_file):
        if not f.exists():
            print(f"ERROR: {f} not found -- {hint}")
            sys.exit(1)

    netlist = sim_netlist(netlist)
    tb_file = str(Path(TB_DIR) / "sdf_checker.sv")
    if not Path(tb_file).exists():
        print(f"ERROR: {tb_file} not found -- copy starter/tb/sdf_checker.sv there")
        sys.exit(1)

    r = run([VLOG_CMD, "-sv", LIB_CELLS] + PROVIDED_RTL + [str(netlist), tb_file],
            desc="vlog SDF compile", check=False, capture=True)
    if r.returncode != 0:
        print("  COMPILE FAIL")
        print(r.stdout[-2000:])
        sys.exit(1)

    # Post-layout: the clock reaches the flip-flops through the clock tree. Tell the checker its average
    # insertion delay (slow corner), so it drives and samples relative to the clock the flops see.
    plus = [f"+sdf_file={sdf_file}"]
    if stage == "pnr":
        lat = clock_latency(tag)
        if lat is not None:
            plus.append(f"+clk_latency_ns={lat:.3f}")
    r = run([VSIM_CMD, "-batch"] + plus + ["-voptargs=+acc",
             "-do", "run -all; quit -f", f"work.{TOP}_sdf_checker"],
            check=False, capture=True)
    out = r.stdout + r.stderr
    print(out[-4000:])
    passed = "ALL PASS" in out or "timing closure confirmed" in out
    x_fail = "TIMING FAIL" in out or "TIMING VIOLATIONS" in out
    print(f"  result: {'PASS' if passed else ('TIMING VIOLATION' if x_fail else 'FAIL')}")


def _genus_power(tag, db_file):
    if not db_file.exists():
        print(f"  WARNING: {db_file} not found -- skipping Genus power")
        return False
    r = subprocess.run(
        [GENUS_CMD, "-batch",
         "-execute", f"set RUN_NAME {tag}",
         "-execute", f"read_db {db_file}",
         "-files", POWER_TCL],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    combined = r.stdout + r.stderr
    if r.returncode != 0 or any(e in combined for e in _LICENSE_ERRORS):
        snippet = combined[:800].strip()
        if snippet:
            print("  Genus failed:\n    " + "\n    ".join(snippet.splitlines()[:6]))
        return False
    print(r.stdout[-2000:])
    return True


def cmd_power(args):
    tag = args.run
    print(f"=== Power analysis ({tag} MHz) ===")
    vcd_file = Path(OUT_DIR) / f"{TOP}_{tag}.vcd"
    saif_file = Path(OUT_DIR) / f"{TOP}_{tag}.saif"
    db_file = Path(OUT_DIR) / f"{TOP}_synth_{tag}.dat"
    if not vcd_file.exists():
        print(f"ERROR: {vcd_file} not found -- run 'bob.py sim --run {tag}' first")
        sys.exit(1)

    print("  VCD -> SAIF ...")
    run(f"csh {VCD2SAIF} {vcd_file} {saif_file}", desc="vcd2saif")
    if _genus_power(tag, db_file):
        print("  Genus power complete.")
    else:
        print("  Genus power unavailable (license server may be down).")
        print(f"  Re-run later: python3 bob.py power --run {tag}")
    rpt = Path(RPT_DIR) / f"{TOP}_{tag}_saif_power.rpt"
    if rpt.exists():
        print(f"power report -> {rpt}")


def cmd_pnr(args):
    tag, stage = args.run, args.stage
    print(f"=== Place and route: {tag} MHz, stage {stage} (Innovus) ===")
    check_tool(INNOVUS_CMD)
    if not Path(PNR_TCL).exists():
        print(f"ERROR: {PNR_TCL} not found -- copy starter/scripts/pnr.tcl there "
              f"and fill in its TODOs")
        sys.exit(1)
    for f in (Path(OUT_DIR) / f"{TOP}_{tag}_map.sv", Path(OUT_DIR) / f"{TOP}_{tag}_map.sdc"):
        if not f.exists():
            print(f"ERROR: {f} not found -- run synth first")
            sys.exit(1)
    for d in (PNR_OUT, PNR_RPT, PNR_LOG):
        Path(d).mkdir(parents=True, exist_ok=True)

    # a single stage starts from the checkpoint the stage before it saved
    if stage not in ("all", "import"):
        prev = PNR_STAGES[PNR_STAGES.index(stage) - 1]
        ckpt = Path(PNR_OUT) / "checkpoints" / f"{TOP}_{tag}_{prev}.enc.dat"
        if not ckpt.exists():
            print(f"ERROR: {ckpt} not found -- run '--stage {prev}' (or '--stage all') first")
            sys.exit(1)

    # same as synth: set RUN_TAG (and PNR_STAGE), then source the student script
    wrapper = Path(PNR_LOG) / f"run_pnr_{tag}_{stage}.tcl"
    wrapper.write_text(f"set RUN_TAG {tag}\nset PNR_STAGE {stage}\nsource {PNR_TCL}\nexit\n")
    log_path = Path(PNR_LOG) / f"innovus_{tag}_{stage}.log"
    print(f"  logging to {log_path}")
    r = subprocess.run(
        [INNOVUS_CMD, "-batch", "-no_gui", "-overwrite", "-files", str(wrapper)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
    log_path.write_text(r.stdout)
    sweep(PNR_LOG, INNOVUS_FILES)
    print(r.stdout[-5000:] if len(r.stdout) > 5000 else r.stdout)
    # Innovus exits 0 even when a Tcl error stops the script: look for that message too
    stopped = "script processing was stopped" in r.stdout
    if r.returncode != 0 or stopped:
        print(f"ERROR: innovus failed ({'the script stopped on an error, see ' + str(log_path) if stopped else f'exit {r.returncode}'})")
        sys.exit(1)
    print(f"reports -> {PNR_RPT}/")
    print(f"logs    -> {PNR_LOG}/")
    print(f"outputs -> {PNR_OUT}/")


# Files and folders the tools create in the project root
TOOL_JUNK = [
    "genus.cmd*", "genus.log*", "genus_*.cmd*", "genus_*.log*", "fv", "rc_model.bin",
    "innovus.cmd*", "innovus.log*", "innovus*.logv*", "timingReports", "scheduling_file.cts*",
    ".cadence", "*.rpt.old", "extLogDir", "*.ccopt*", "rcOut", "clock_tree_debug*",
    "work", "transcript", "*.wlf", "*.vstf", "modelsim.ini", "__pycache__", ".*.slk.gz",
]
GENERATED = [RPT_DIR, LOG_DIR, OUT_DIR, PNR_OUT, PNR_RPT, PNR_LOG]


def cmd_clean(args):
    import shutil
    targets = sorted({p for pat in TOOL_JUNK for p in glob.glob(pat)})
    targets += sorted(glob.glob("**/.DS_Store", recursive=True))
    if args.all:
        found = [d for d in GENERATED if Path(d).exists()]
        if found:
            print("--all also deletes: " + ", ".join(found))
            if input("Delete these generated reports and outputs? [y/N] ").strip().lower() == "y":
                targets += found
    if not targets:
        print("nothing to clean")
        return
    for t in targets:
        shutil.rmtree(t) if Path(t).is_dir() else os.remove(t)
        print(f"  removed {t}")
    # keep the empty submission folders (each holds a .gitkeep)
    for d in GENERATED + [os.path.join(PNR_OUT, "checkpoints")]:
        if d in targets or os.path.dirname(d) in targets:
            Path(d).mkdir(parents=True, exist_ok=True)
            (Path(d) / ".gitkeep").touch()
    print(f"cleaned {len(targets)} item(s)")


def main():
    if sys.argv[1:2] != ["clean"]:
        _load_cmc_env()
    p = argparse.ArgumentParser(description="ELEC 402 P3 runner")
    sub = p.add_subparsers(dest="command")
    def with_run(name, help_):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--run", choices=TAGS, default=TAGS[0], help="target frequency (MHz)")
        return sp
    with_run("check", "RTL functional sim")
    with_run("synth", "synthesis")
    pt = with_run("timing", "SDF-annotated gate-level timing check")
    pt.add_argument("--stage", choices=["syn", "pnr"], default="syn",
                    help="syn: synthesized netlist, pnr: post-layout netlist")
    with_run("sim", "gate-level sim (no SDF) -> VCD for power")
    with_run("power", "VCD -> SAIF -> power report")
    pn = with_run("pnr", "place and route (Innovus)")
    pn.add_argument("--stage", choices=["all"] + PNR_STAGES, default="all",
                    help="all stages (default), or one stage, starting from the previous stage's checkpoint")
    pcl = sub.add_parser("clean", help="delete files the tools leave behind")
    pcl.add_argument("--all", action="store_true", help="also delete generated reports and outputs")

    args = p.parse_args()
    {
        "check":  cmd_check,
        "synth":  cmd_synth,
        "timing": cmd_timing,
        "sim":    cmd_sim,
        "power":  cmd_power,
        "pnr":    cmd_pnr,
        "clean":  cmd_clean,
    }.get(args.command, lambda _: p.print_help())(args)


if __name__ == "__main__":
    main()
