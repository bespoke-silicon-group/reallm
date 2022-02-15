// Calculate inputs cycles

`include "bsg_defines.v"

module inputs_cycles_calculate #(
   parameter `BSG_INV_PARAM(id_width_p)
  ,parameter `BSG_INV_PARAM(size_width_p)
  ,parameter data_bytes_p = 2 // bytes / data

  ,parameter `BSG_INV_PARAM(num_in_p)

  // chiplet hardware settings
  ,parameter `BSG_INV_PARAM(num_macs_p) // macs / cycle
  // chiplet configuration
  ,parameter `BSG_INV_PARAM(inputs_select_p) // -1: gather, 0~num_in_p-1: select
  ,parameter `BSG_INV_PARAM(macs_per_data_p) // to perform # macs per input, also generate # output

  ,parameter width_p = id_width_p + size_width_p
  ,parameter cycles_width_p = size_width_p + `BSG_SAFE_CLOG2(macs_per_data_p) - `BSG_SAFE_CLOG2(num_macs_p)
)
  ( input  [num_in_p-1:0][width_p-1:0] data_i
   ,output [width_p-1:0]               data_o
   ,output [cycles_width_p-1:0]        cycle_o
  );

  logic [size_width_p-1:0] workload_size;

  if (num_in_p == 1) begin: single_input
    assign workload_size = data_i[0][size_width_p-1:0];
    assign data_o = data_i;
  end
  else begin: multi_inputs
    wire [id_width_p-1:0] workload_id = data_i[0][width_p-1:size_width_p];
    if (inputs_select_p == -1) begin: add_all_inputs
      logic [num_in_p:0][size_width_p-1:0] mid_results;
      logic [num_in_p-1:0] overflow;
      genvar i;
      for (i = 0; i < num_in_p; i=i+1) begin: inputs_add
        bsg_adder_ripple_carry #(.width_p(size_width_p)) adder
           (.a_i (data_i[i][size_width_p-1:0])
           ,.b_i (mid_results[i])
           ,.c_o (overflow[i])
           ,.s_o (mid_results[i+1])
           );
      end
      assign mid_results[0] = '0;
      assign workload_size = mid_results[num_in_p];
    end
    else if (inputs_select_p < num_in_p) begin: select_one_input 
      assign workload_size = data_i[inputs_select_p][size_width_p-1:0];
    end
    else begin
      $error("inputs_select_p wrong!\n");
    end
    assign data_o = {workload_id, workload_size};
  end

  assign cycle_o = workload_size * macs_per_data_p / num_macs_p;


endmodule
