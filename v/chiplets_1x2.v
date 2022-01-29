// Chiplet

`include "bsg_defines.v"

module chiplets_1x2 #(
   parameter `BSG_INV_PARAM(id_width_p)
  ,parameter `BSG_INV_PARAM(size_width_p)
  ,parameter data_bytes_p = 2 // bytes / data

  // chiplet hardware settings
  ,parameter `BSG_INV_PARAM(num_macs_p) // macs / cycle
  ,parameter `BSG_INV_PARAM(bandwidth_p) // bytes / cycle
  ,parameter width_p = id_width_p + size_width_p
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

  logic c_1_1_v_o, c_1_1_ready_i;
  logic [width_p-1:0] c_1_1_data_o;
  chiplet #(
     .id_width_p(id_width_p)
    ,.size_width_p(size_width_p)
    ,.data_bytes_p(data_bytes_p)
    ,.num_macs_p(num_macs_p)
    ,.num_in_p(1)
    ,.num_out_p(1)
    ,.inputs_select_p(0)
    ,.outputs_config_p(0)
    ,.macs_per_data_p(4)
  ) chiplet_1_1
      ( .clk_i(clk_i)
      , .reset_i(reset_i)
  
      // input side
      , .ready_o(ready_o)
      , .data_i(data_i)
      , .v_i(v_i)    
  
      // output side
      , .v_o(c_1_1_v_o)
      , .data_o(c_1_1_data_o) 
      , .ready_i(c_1_1_ready_i)
      );

  logic l_1_v_o, l_1_ready_i;
  logic [width_p-1:0] l_1_data_o;
  link #(
     .id_width_p(id_width_p)
    ,.size_width_p(size_width_p)
    ,.data_bytes_p(data_bytes_p)
    ,.bandwidth_p(bandwidth_p)
  ) link_1
      ( .clk_i(clk_i)
      , .reset_i(reset_i)
  
      // input side
      , .ready_o(c_1_1_ready_i)
      , .data_i(c_1_1_data_o)
      , .v_i(c_1_1_v_o)    
  
      // output side
      , .v_o(l_1_v_o)
      , .data_o(l_1_data_o) 
      , .ready_i(l_1_ready_i)
      );

  chiplet #(
     .id_width_p(id_width_p)
    ,.size_width_p(size_width_p)
    ,.data_bytes_p(data_bytes_p)
    ,.num_macs_p(num_macs_p)
    ,.num_in_p(1)
    ,.num_out_p(1)
    ,.inputs_select_p(0)
    ,.outputs_config_p(0)
    ,.macs_per_data_p(2)
  ) chiplet_1_2
      ( .clk_i(clk_i)
      , .reset_i(reset_i)
  
      // input side
      , .ready_o(l_1_ready_i)
      , .data_i(l_1_data_o)
      , .v_i(l_1_v_o)    
  
      // output side
      , .v_o(v_o)
      , .data_o(data_o) 
      , .ready_i(ready_i)
      );
endmodule

