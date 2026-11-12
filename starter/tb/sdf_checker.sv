// sdf_checker.sv -- ELEC 402 2026 Project 3. Top: mkMACBuff_sdf_checker. Copy to submission/testbench/.
// bob.py passes the synthesized (timing --stage syn) or post-layout (timing --stage pnr) SDF with +sdf_file=...
// An X on memVal_data after VALID_memVal means a timing violation. Clock fixed at 2 ns (500 MHz).

`timescale 1ns/1ps

module mkMACBuff_sdf_checker;

    localparam real HALF_NS = 1.0;
    localparam real DRIVE_NS = 0.5;  // input delay: 1/4 of the 2 ns period
    localparam real SAMPLE_NS = 1.5; // output delay: outputs must be valid 1/4 period before the next edge
    localparam int  N = 1; // TODO: the same N as your mkMACBuff

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

    memory_wrapper_2port #(.DEPTH(64), .LOGDEPTH(6), .WIDTH(N), .MEMTYPE(0)) MEM (
        .clkA(clk), .aA(readMem_addr), .cenA(~EN_readMem), .q(readMem_val),
        .clkB(clk), .aB(writeMem_addr), .cenB(~EN_writeMem), .d(writeMem_val)
    );

    initial begin
        clk = 0;
        #(HALF_NS * 1ns);
        forever #(HALF_NS * 1ns) clk = ~clk;
    end

    initial begin
        string sdf_path;
        if ($value$plusargs("sdf_file=%s", sdf_path)) begin
            $sdf_annotate(sdf_path, DUT, , , "MAXIMUM");
            $display("# [SDF] annotated: %s", sdf_path);
        end else
            $display("# [SDF WARNING] +sdf_file not provided");
    end

    initial begin
        // TODO: fill/drain the memory under SDF annotation; flag any X on memVal_data as a
        //       timing violation; print "ALL PASS" on success.
        $display("TODO: implement sdf_checker");
        $finish;
    end

endmodule : mkMACBuff_sdf_checker
