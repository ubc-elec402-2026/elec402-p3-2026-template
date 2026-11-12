// mkMACBuff.sv -- ELEC 402 2026 Project 3: signed 4-element dot product with ReLU and a fill/drain buffer.
// Copy to submission/rtl/. Each accepted beat (EN_mac & RDY_mac) stores one result in the 64-entry memory;
// EN_blockRead & RDY_blockRead drain all 64 in order. Active-low synchronous reset (RST_N), any pipeline depth OK.

// TODO: set N to the result width you derive (README "1. The dot product"), and justify it in your report.
module mkMACBuff #(parameter N = 1) (
    input logic clk,
    input logic RST_N, // active-low synchronous reset

    // MAC / write side
    input logic EN_mac,
    input logic [15:0] mac_a0, mac_a1, mac_a2, mac_a3, // row of A (signed)
    input logic [15:0] mac_b0, mac_b1, mac_b2, mac_b3, // column of B (signed)
    input logic relu,
    output logic RDY_mac,

    // Read / drain side
    input logic EN_blockRead,
    output logic RDY_blockRead,
    output logic VALID_memVal,
    output logic [N-1:0] memVal_data,

    // Memory write port
    output logic EN_writeMem,
    output logic [5:0] writeMem_addr,
    output logic [N-1:0] writeMem_val,

    // Memory read port
    output logic EN_readMem,
    output logic [5:0] readMem_addr,
    input logic [N-1:0] readMem_val
);

    // TODO: implement the fill/drain control logic and instantiate your four
    //       hand-pipelined signed multipliers and your adder tree here.

endmodule : mkMACBuff
