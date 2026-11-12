# saif2power.tcl - ELEC 402 Project 3 power report from the SAIF file. Run by bob.py power.

set TOP_LEVEL mkMACBuff
set RPT_DIR   ./submission/syn_reports

set SAIF_FILE "./submission/outputs/${TOP_LEVEL}_${RUN_NAME}.saif"
set RPT_FILE  "${RPT_DIR}/${TOP_LEVEL}_${RUN_NAME}_saif_power.rpt"

sh mkdir -p $RPT_DIR

read_saif -instance mkMACBuff_tb/DUT -scale 1 $SAIF_FILE

report_power > $RPT_FILE
puts "SAIF-annotated power report: $RPT_FILE"
