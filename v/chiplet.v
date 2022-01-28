// Chiplet

`include "bsg_defines.v"

module chiplet #(
   parameter `BSG_INV_PARAM(id_width_p)
  ,parameter `BSG_INV_PARAM(size_width_p)
  ,parameter data_bytes_p = 2 // bytes / data

  // chiplet hardware settings
  ,parameter `BSG_INV_PARAM(num_macs_p) // macs / cycle
  // chiplet configuration
  ,parameter `BSG_INV_PARAM(macs_per_data_p) // to perform # macs per input, also generate # output

  ,parameter width_p = id_width_p + size_width_p
  ,parameter cycles_width_p = size_width_p + `BSG_SAFE_CLOG2(macs_per_data_p) - `BSG_SAFE_CLOG2(num_macs_p)
)
  ( input clk_i
    , input reset_i

    // input side
    , output              ready_o 
    , input [width_p-1:0] data_i 
    , input               v_i     

    // output side
    , output              v_o   
    , output[width_p-1:0] data_o
    , input               ready_i 
    );

  logic [width_p-1:0] data_to_ctr, data_from_ctr;
  logic yumi_from_ctr, valid_to_ctr, valid_from_ctr, ready_to_ctr;

  bsg_two_fifo #(.width_p(width_p)) input_fifo
  (  .clk_i   (clk_i)
    ,.reset_i (reset_i)
    // input side
    ,.ready_o (ready_o)
    ,.data_i  (data_i)
    ,.v_i     (v_i)    
    // output side
    ,.v_o     (valid_to_ctr)
    ,.data_o  (data_to_ctr) 
    ,.yumi_i  (yumi_from_ctr)
    );

  wire [id_width_p-1:0] workload_id = data_to_ctr[width_p-1:size_width_p];
  wire [size_width_p-1:0] workload_size = data_to_ctr[size_width_p-1:0];

  wire [cycles_width_p-1:0] cycles = workload_size * macs_per_data_p / num_macs_p;

  counter #(
    .cycles_width_p(cycles_width_p)
   ,.width_p       (width_p)
  ) counter_inst
  (  .clk_i    (clk_i)
    ,.reset_i  (reset_i)
    // input side
    ,.yumi_o   (yumi_from_ctr)
    ,.data_i   (data_to_ctr)
    ,.cycles_i (cycles)
    ,.v_i      (valid_to_ctr)    
    // output side
    ,.v_o      (valid_from_ctr)
    ,.data_o   (data_from_ctr) 
    ,.ready_i  (ready_to_ctr)
  );
  
  logic [width_p-1:0] data;
  wire yumi = ready_i & v_o;
  bsg_two_fifo #(.width_p(width_p),.allow_enq_deq_on_full_p(1)) output_fifo
  (  .clk_i   (clk_i)
    ,.reset_i (reset_i)
    // input side
    ,.ready_o (ready_to_ctr)
    ,.data_i  (data_from_ctr)
    ,.v_i     (valid_from_ctr)    
    // output side
    ,.v_o     (v_o)
    ,.data_o  (data) 
    ,.yumi_i  (yumi)
    );

  assign data_o = {data[width_p-1 -: id_width_p], size_width_p'(macs_per_data_p)};


endmodule

