# pnr.tcl - Cadence Innovus place and route template, ELEC 402 2026 Project 3
# Copy to submission/scripts/pnr.tcl, fill in every TODO, run with: python3 bob.py pnr --run 500 [--stage cts]
# Run synthesis first. Do not rename the output files: bob.py and the SDF check look for them by name.

set TOP     mkMACBuff
set SYN     ./submission/outputs
set OUT     ./submission/pnr_outputs
set RPT     ./submission/pnr_reports
set TAG     500
set PERIOD  2.0

set KIT      /ubc/ece/data/cmc2/kits/GPDK45/gsclib045_all_v4.4/gsclib045
set TECH_LEF $KIT/lef/gsclib045_tech.lef
set CELL_LEF $KIT/lef/gsclib045_macro.lef
set LIB_SLOW $KIT/timing/slow_vdd1v0_basicCells.lib
set LIB_FAST $KIT/timing/fast_vdd1v0_basicCells.lib
set QRC      $KIT/qrc/qx/gpdk045.tch
set FILLERS  {FILL64 FILL32 FILL16 FILL8 FILL4 FILL2 FILL1}

file mkdir $OUT $RPT
namespace eval pnr {}

proc pnr::import {} {
    global TOP SYN OUT TECH_LEF CELL_LEF LIB_SLOW LIB_FAST QRC TAG
    global init_verilog init_top_cell init_lef_file init_mmmc_file init_pwr_net init_gnd_net
    set fin [open $SYN/${TOP}_${TAG}_map.sdc r]
    set fout [open $OUT/${TOP}_${TAG}_cts.sdc w]
    puts -nonewline $fout [regsub -all -line {^\s*set_clock_(latency|uncertainty)\M.*$} [read $fin] ""]
    close $fin; close $fout

    set f [open $OUT/mmmc.tcl w]
    puts $f "create_library_set -name slow_set -timing {$LIB_SLOW}"
    puts $f "create_library_set -name fast_set -timing {$LIB_FAST}"
    puts $f "create_rc_corner -name rc_typ -qx_tech_file {$QRC}"
    puts $f "create_delay_corner -name slow_dc -library_set slow_set -rc_corner rc_typ"
    puts $f "create_delay_corner -name fast_dc -library_set fast_set -rc_corner rc_typ"
    puts $f "create_constraint_mode -name func -sdc_files {$SYN/${TOP}_${TAG}_map.sdc}"
    puts $f "create_analysis_view -name setup_view -constraint_mode func -delay_corner slow_dc"
    puts $f "create_analysis_view -name hold_view -constraint_mode func -delay_corner fast_dc"
    puts $f "set_analysis_view -setup {setup_view} -hold {hold_view}"
    close $f

    set init_verilog   $SYN/${TOP}_${TAG}_map.sv
    set init_top_cell  $TOP
    set init_lef_file  [list $TECH_LEF $CELL_LEF]
    set init_mmmc_file $OUT/mmmc.tcl
    set init_pwr_net   VDD
    set init_gnd_net   VSS
    init_design

    globalNetConnect VDD -type pgpin -pin VDD -inst * -override
    globalNetConnect VSS -type pgpin -pin VSS -inst * -override
    globalNetConnect VDD -type tiehi -inst * -override
    globalNetConnect VSS -type tielo -inst * -override
}

proc pnr::floorplan {{util 0.70}} {
    #### TODO 1: floorPlan, aspect ratio 1, utilization $util, 8 um core-to-die spacing on every edge
}

proc pnr::power {} {
    #### TODO 2: core ring on Metal7 / Metal8, 1.8 um wide (addRing)
    #### TODO 3: power stripes (addStripe), then connect the cell rails (sroute)
}

proc pnr::pins {} {
    #### TODO 4: every pin on Metal3 (editPin), spread over the middle 80% of each edge.
    ####         Left, top to bottom: mac interface, clk + RST_N in the centre, blockRead, memVal.
    ####         Right, top to bottom: writeMem interface, then readMem interface. See README "Pins".
}

proc pnr::place {} {
    global RPT
    #### TODO 5: place, optimize before CTS, then timeDesign -preCTS -outDir $RPT/preCTS.
    ####         Not within 100 ps of meeting setup? Lower the utilization.
}

proc pnr::cts {} {
    global TOP OUT RPT PERIOD TAG
    update_constraint_mode -name func -sdc_files $OUT/${TOP}_${TAG}_cts.sdc
    #### TODO 6: clock tree on Metal4 to Metal7, max transition $PERIOD / 8, max fanout 4.
    ####         Build it (ccopt_design), then fix setup and hold.
    report_ccopt_clock_trees -file $RPT/${TOP}_${TAG}_clock_trees.rpt
    report_ccopt_skew_groups -file $RPT/${TOP}_${TAG}_skew_groups.rpt
    timeDesign -postCTS -outDir $RPT/postCTS
}

proc pnr::route {} {
    global RPT
    #### TODO 7: route up to Metal10, timing and SI driven, fix antennas, then fix setup and hold.
    timeDesign -postRoute -outDir $RPT/postRoute
    timeDesign -postRoute -hold -outDir $RPT/postRoute_hold
}

proc pnr::fill {} {
    global TOP RPT FILLERS TAG
    #### TODO 8: fill the core gaps with $FILLERS, then check for gaps of 0.2 um and up
    setDesignMode -bottomRoutingLayer 1
    setNanoRouteMode -routeBottomRoutingLayer 1
    ecoRoute
}

proc pnr::verify {} {
    global TOP RPT TAG
    verify_drc -report $RPT/${TOP}_${TAG}_drc.rpt
    verifyConnectivity -report $RPT/${TOP}_${TAG}_connectivity.rpt
    verifyProcessAntenna -report $RPT/${TOP}_${TAG}_antenna.rpt
}

proc pnr::finish {} {
    global TOP OUT RPT TAG
    report_area > $RPT/${TOP}_${TAG}_area.rpt
    report_timing -late > $RPT/${TOP}_${TAG}_setup.rpt
    report_timing -early > $RPT/${TOP}_${TAG}_hold.rpt
    report_power -outfile $RPT/${TOP}_${TAG}_power.rpt
    summaryReport -noHtml -outfile $RPT/${TOP}_${TAG}_summary.rpt
    saveNetlist $OUT/${TOP}_${TAG}_pnr.v
    write_sdf -view setup_view $OUT/${TOP}_${TAG}_pnr.sdf
    saveDesign $OUT/${TOP}_${TAG}_pnr.enc
}

proc pnr::save {stage} {
    global TOP OUT TAG
    file mkdir $OUT/checkpoints
    saveDesign $OUT/checkpoints/${TOP}_${TAG}_${stage}.enc
}

proc pnr::load {stage} {
    global TOP OUT TAG
    restoreDesign $OUT/checkpoints/${TOP}_${TAG}_${stage}.enc.dat $TOP
}

proc pnr_run {tag period util {stage all}} {
    global TAG PERIOD
    set TAG $tag
    set PERIOD $period
    set order {import floorplan power pins place cts route fill verify finish}

    if {$stage eq "all"} {
        pnr::import;          pnr::save import
        pnr::floorplan $util; pnr::save floorplan
        pnr::power;           pnr::save power
        pnr::pins;            pnr::save pins
        pnr::place;           pnr::save place
        pnr::cts;             pnr::save cts
        pnr::route;           pnr::save route
        pnr::fill;            pnr::save fill
        pnr::verify;          pnr::save verify
        pnr::finish
    } else {
        set i [lsearch -exact $order $stage]
        if {$i < 0} { error "unknown stage '$stage': use all or one of $order" }
        if {$i > 0} { pnr::load [lindex $order [expr {$i - 1}]] }
        if {$stage eq "floorplan"} { pnr::floorplan $util } else { pnr::$stage }
        pnr::save $stage
    }
    puts "done: $tag ($stage)"
}

if {[info exists STAGES_ONLY]} { return }

if {![info exists RUN_TAG]} { set RUN_TAG "all" }
if {![info exists PNR_STAGE]} { set PNR_STAGE "all" }

#### TODO 9: call pnr_run with the tag, the period (2.0 ns), your utilization and $PNR_STAGE.
if {$RUN_TAG eq "500" || $RUN_TAG eq "all"} {
}

puts "place and route complete - reports in $RPT"
