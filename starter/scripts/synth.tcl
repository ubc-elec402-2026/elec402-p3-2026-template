# synth.tcl - Cadence Genus synthesis template, ELEC 402 2026 Project 3
# Copy to submission/scripts/synth.tcl, fill in every TODO, run with: python3 bob.py synth --run 500
# Do not rename the output files: bob.py, the SDF check and place and route look for them by name.

set TOP_LEVEL    mkMACBuff
set RTL_DIR      ./submission/rtl
set STR_DIR      ./starter/rtl
set RPT_DIR      ./submission/syn_reports
set OUT_DIR      ./submission/outputs
set LIB_PATH     /ubc/ece/data/cmc2/kits/GPDK45/gsclib045_all_v4.4/gsclib045/timing

sh mkdir -p $RPT_DIR $OUT_DIR

set_db lib_search_path [concat [get_db lib_search_path] $LIB_PATH]
set_db library "slow_vdd1v0_basicCells.lib"

#### TODO 0: add the other arguments synth_run needs (for example the synthesis and reporting periods).
proc synth_run {tag} {
    global TOP_LEVEL RTL_DIR STR_DIR RPT_DIR OUT_DIR

    read_hdl -sv [concat $STR_DIR/memory_wrapper_2port.sv $STR_DIR/registerArray.sv [glob $RTL_DIR/*.sv]]
    elaborate $TOP_LEVEL
    set cur [lindex [get_db designs .name] end]
    current_design $cur

    #### TODO 1: clock, clock latency and clock uncertainty (see README "Synthesis > Constraints")

    #### TODO 2: input drive and output load

    #### TODO 3: input and output delays, and the max fanout of the input nets

    #### TODO 4: synthesize and optimize. Retiming may only rebalance YOUR pipeline. See README "Design requirements > 2".

    #### TODO 5: re-define the clock at the graded period for the reports.
    ####         Re-creating the clock drops every constraint that refers to it: apply them again.

    report_area   > $RPT_DIR/${TOP_LEVEL}_${tag}_area.rpt
    report_gates  > $RPT_DIR/${TOP_LEVEL}_${tag}_gates.rpt
    report_timing > $RPT_DIR/${TOP_LEVEL}_${tag}_timing.rpt
    report_timing -from [all_inputs] > $RPT_DIR/${TOP_LEVEL}_${tag}_timing_in.rpt
    report_timing -to [all_outputs]  > $RPT_DIR/${TOP_LEVEL}_${tag}_timing_out.rpt
    report_power  > $RPT_DIR/${TOP_LEVEL}_${tag}_power.rpt
    report_qor    > $RPT_DIR/${TOP_LEVEL}_${tag}_qor.rpt

    write_hdl -mapped > $OUT_DIR/${TOP_LEVEL}_${tag}_map.sv
    write_sdf         > $OUT_DIR/${TOP_LEVEL}_${tag}_map.sdf
    write_sdc         > $OUT_DIR/${TOP_LEVEL}_${tag}_map.sdc
    write_db -to        $OUT_DIR/${TOP_LEVEL}_synth_${tag}.dat

    puts "done: $tag"
}

if {![info exists RUN_TAG]} { set RUN_TAG "all" }

#### TODO 6: call synth_run. Reports are graded at 2.0 ns (500 MHz).
if {$RUN_TAG eq "500" || $RUN_TAG eq "all"} {
}

puts "synthesis complete - reports in $RPT_DIR"
