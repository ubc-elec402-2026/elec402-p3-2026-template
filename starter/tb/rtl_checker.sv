// rtl_checker.sv -- ELEC 402 2026 Project 3. Top: mkMACBuff_tb. Copy to submission/testbench/.
// The harness below wires the DUT to the memory and makes a 500 MHz clock (CLK_HALF_NS 1.0).
// Drive fill/drain, scoreboard every result read out. See README "Testbench requirements" for what to cover.

`timescale 1ns/1ps

module mkMACBuff_tb #(
    parameter real CLK_HALF_NS = 1.0, // 2 ns period -> 500 MHz
    parameter int  N = 1 // TODO: the same N as your mkMACBuff
);

    logic clk, RST_N;

    logic EN_mac, relu;
    logic [15:0] mac_a0, mac_a1, mac_a2, mac_a3;
    logic [15:0] mac_b0, mac_b1, mac_b2, mac_b3;
    logic RDY_mac;

    logic EN_blockRead;
    logic RDY_blockRead;
    logic VALID_memVal;
    logic [N-1:0] memVal_data;

    logic EN_writeMem, EN_readMem;
    logic [5:0] writeMem_addr, readMem_addr;
    logic [N-1:0] writeMem_val, readMem_val;

    mkMACBuff #(.N(N)) DUT (
        .clk(clk), .RST_N(RST_N),
        .EN_mac(EN_mac),
        .mac_a0(mac_a0), .mac_a1(mac_a1), .mac_a2(mac_a2), .mac_a3(mac_a3),
        .mac_b0(mac_b0), .mac_b1(mac_b1), .mac_b2(mac_b2), .mac_b3(mac_b3),
        .relu(relu), .RDY_mac(RDY_mac),
        .EN_blockRead(EN_blockRead), .RDY_blockRead(RDY_blockRead),
        .VALID_memVal(VALID_memVal), .memVal_data(memVal_data),
        .EN_writeMem(EN_writeMem), .writeMem_addr(writeMem_addr),
        .writeMem_val(writeMem_val),
        .EN_readMem(EN_readMem), .readMem_addr(readMem_addr),
        .readMem_val(readMem_val)
    );

    memory_wrapper_2port #(
        .DEPTH(64), .LOGDEPTH(6), .WIDTH(N), .MEMTYPE(0)
    ) MEM (
        .clkA(clk), .aA(readMem_addr), .cenA(~EN_readMem), .q(readMem_val),
        .clkB(clk), .aB(writeMem_addr), .cenB(~EN_writeMem), .d(writeMem_val)
    );

    initial clk = 0;
    always #(CLK_HALF_NS * 1ns) clk = ~clk;

    initial begin
        // TODO: reset, then at least 9 matrix multiplies C (8x8) = A (8x4) x B (4x8) on this one DUT
        //       (ReLU off and on): send the beats, read C back, compare with A x B computed here,
        //       print "ALL PASS" on success. bob.py greps for "ALL PASS".
        $display("TODO: implement rtl_checker");
        $finish;
    end

endmodule
