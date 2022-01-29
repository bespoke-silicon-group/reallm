// Inputs gather

`include "bsg_defines.v"

module inputs_gather #(
   parameter `BSG_INV_PARAM(width_p)
  ,parameter `BSG_INV_PARAM(num_in_p)
)
  ( input clk_i
    , input reset_i

    // input side
    , output [num_in_p-1:0]               ready_o 
    , input  [num_in_p-1:0] [width_p-1:0] data_i
    , input  [num_in_p-1:0]               v_i     

    // output side
    , output                              v_o   
    , output [num_in_p-1:0] [width_p-1:0] data_o
    , input                               yumi_i 
    );

  // Multiple input FIFOs
  logic [num_in_p-1:0] in_fifos_valid;
  genvar i;
  for (i = 0; i < num_in_p; i=i+1) begin: in_fifos
    bsg_two_fifo #(.width_p(width_p)) input_fifo
    (  .clk_i   (clk_i)
      ,.reset_i (reset_i)
      // input side
      ,.ready_o (ready_o[i])
      ,.data_i  (data_i[i])
      ,.v_i     (v_i[i])    
      // output side
      ,.v_o     (in_fifos_valid[i])
      ,.data_o  (data_o[i]) 
      ,.yumi_i  (yumi_i)
      );
  end

  bsg_reduce #(.width_p(num_in_p)
              ,.and_p (1)
  ) valid_and_reduce
  (  .i(in_fifos_valid)
    ,.o(v_o)
  );

endmodule
