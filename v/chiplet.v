// Chiplet

`include "bsg_defines.v"

module chiplet #(
   parameter `BSG_INV_PARAM(id_width_p)
  ,parameter `BSG_INV_PARAM(size_width_p)
  ,parameter data_bytes_p = 2 // bytes / data

  ,parameter `BSG_INV_PARAM(num_in_p)
  ,parameter `BSG_INV_PARAM(num_out_p)

  // chiplet hardware settings
  ,parameter `BSG_INV_PARAM(num_macs_p) // macs / cycle
  // chiplet configuration
  ,parameter `BSG_INV_PARAM(inputs_select_p) // -1: gather, 0~num_in_p-1: select
  ,parameter `BSG_INV_PARAM(outputs_config_p) // -1: gather, 0~num_in_p-1: select
  ,parameter `BSG_INV_PARAM(macs_per_data_p) // to perform # macs per input, also generate # output

  ,parameter width_p = id_width_p + size_width_p
  ,parameter cycles_width_p = size_width_p + `BSG_SAFE_CLOG2(macs_per_data_p) - `BSG_SAFE_CLOG2(num_macs_p)
)
  ( input clk_i
    , input reset_i

    // input side
    , output [num_in_p-1:0]               ready_o 
    , input  [num_in_p-1:0] [width_p-1:0] data_i
    , input  [num_in_p-1:0]               v_i     

    // output side
    , output [num_out_p-1:0]               v_o   
    , output [num_out_p-1:0] [width_p-1:0] data_o
    , input  [num_out_p-1:0]               ready_i 
    );

  // Multiple input gather
  logic inputs_v;
  logic inputs_yumi;
  logic [num_in_p-1:0][width_p-1:0] inputs_data;

  inputs_gather #(.width_p(width_p), .num_in_p(num_in_p)) in
  (  .clk_i
    ,.reset_i
    ,.ready_o (ready_o)
    ,.data_i  (data_i)
    ,.v_i     (v_i)
    ,.v_o     (inputs_v)
    ,.data_o  (inputs_data)
    ,.yumi_i  (inputs_yumi)
  );

  logic [width_p-1:0] data_to_ctr;
  logic [cycles_width_p-1:0] cycles;
  inputs_cycles_calculate #(
    .id_width_p(id_width_p)
   ,.size_width_p(size_width_p)
   ,.num_in_p(num_in_p)
   ,.num_macs_p(num_macs_p)
   ,.inputs_select_p(inputs_select_p)
   ,.macs_per_data_p(macs_per_data_p)
  ) in_cyc_cal
  ( 
     .data_i  (inputs_data)
    ,.data_o  (data_to_ctr) 
    ,.cycle_o (cycles)
  );

  // Counter
  logic [width_p-1:0] data_from_ctr;
  logic valid_from_ctr, ready_to_ctr;
  counter #(
    .cycles_width_p(cycles_width_p)
   ,.width_p       (width_p)
  ) counter_inst
  (  .clk_i    (clk_i)
    ,.reset_i  (reset_i)
    // input side
    ,.yumi_o   (inputs_yumi)
    ,.data_i   (data_to_ctr)
    ,.cycles_i (cycles)
    ,.v_i      (inputs_v)    
    // output side
    ,.v_o      (valid_from_ctr)
    ,.data_o   (data_from_ctr) 
    ,.ready_i  (ready_to_ctr)
  );

  wire [width_p-1:0] output_data = {data_from_ctr[width_p-1 -: id_width_p], size_width_p'(macs_per_data_p)};
  
  logic [num_out_p-1:0][width_p-1:0] outputs_data;
  outputs_workload_calculate #(
    .id_width_p(id_width_p)
   ,.size_width_p(size_width_p)
   ,.num_out_p(num_out_p)
   ,.outputs_config_p(outputs_config_p)
  ) out_work_cal
  (  .data_i (output_data)
    ,.data_o (outputs_data)
  );

  // Multiple outputs scatter
  wire [num_out_p-1:0] yumi = ready_i & v_o;
  outputs_scatter #(.width_p(width_p), .num_out_p(num_out_p)) out
  (  .clk_i
    ,.reset_i
    ,.ready_o (ready_to_ctr)
    ,.data_i  (outputs_data)
    ,.v_i     (valid_from_ctr)
    ,.v_o     (v_o)
    ,.data_o  (data_o)
    ,.yumi_i  (yumi)
  );


endmodule

