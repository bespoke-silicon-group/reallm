// Outputs scatter

`include "bsg_defines.v"

module outputs_scatter #(
   parameter `BSG_INV_PARAM(width_p)
  ,parameter `BSG_INV_PARAM(num_out_p)
)
  ( input clk_i
    , input reset_i

    // input side
    , output                               ready_o 
    , input  [num_out_p-1:0] [width_p-1:0] data_i
    , input                                v_i     

    // output side
    , output [num_out_p-1:0]               v_o   
    , output [num_out_p-1:0] [width_p-1:0] data_o
    , input  [num_out_p-1:0]               yumi_i 
    );

  // Multiple output FIFOs
  logic [num_out_p-1:0] out_fifos_ready;
  genvar i;
  for (i = 0; i < num_out_p; i=i+1) begin: in_fifos
    bsg_two_fifo #(.width_p(width_p), .allow_enq_deq_on_full_p(1)) output_fifo
    (  .clk_i   (clk_i)
      ,.reset_i (reset_i)
      // input side
      ,.ready_o (out_fifos_ready[i])
      ,.data_i  (data_i[i])
      ,.v_i     (v_i)    
      // output side
      ,.v_o     (v_o[i])
      ,.data_o  (data_o[i]) 
      ,.yumi_i  (yumi_i[i])
      );
  end

  bsg_reduce #(.width_p(num_out_p)
              ,.and_p (1)
  ) ready_and_reduce
  (  .i(out_fifos_ready)
    ,.o(ready_o)
  );

endmodule
