#!/bin/tcsh -f
# vcd2saif_run.csh - ELEC 402 Project 3 VCD to SAIF. Run by bob.py power.

if ($#argv != 2) then
    echo "Usage: $0 <input.vcd> <output.saif>"
    exit 1
endif

set VCD_IN   = $argv[1]
set SAIF_OUT = $argv[2]

if (! -f "$VCD_IN") then
    echo "ERROR: VCD file not found: $VCD_IN"
    exit 1
endif

source /CMC/scripts/synopsys.syn.2022.12.csh

vcd2saif -input "$VCD_IN" -output "$SAIF_OUT"
if ($status != 0) then
    echo "ERROR: vcd2saif failed"
    exit 1
endif

echo "SAIF written to $SAIF_OUT"
