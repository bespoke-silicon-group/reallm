// Calculate outputs workload

`include "bsg_defines.v"

module outputs_workload_calculate #(
   parameter `BSG_INV_PARAM(id_width_p)
  ,parameter `BSG_INV_PARAM(size_width_p)
  ,parameter data_bytes_p = 2 // bytes / data

  ,parameter `BSG_INV_PARAM(num_out_p)

  // chiplet configuration
  ,parameter `BSG_INV_PARAM(outputs_config_p) // 0: duplicate to all outputs; 1: splits to num_out_p outputs evenly

  ,parameter width_p = id_width_p + size_width_p
)
  ( input  [width_p-1:0]                data_i
   ,output [num_out_p-1:0][width_p-1:0] data_o
  );


  logic [id_width_p-1:0] workload_id;
  logic [size_width_p-1:0] workload_size;

  assign workload_id = data_i[width_p-1:size_width_p];
  if (num_out_p == 1) begin: single_output
    assign data_o[0] = data_i;
    assign workload_size = data_i[size_width_p-1:0];
  end
  else begin: multi_outputs

    if (outputs_config_p == 0) begin: replicate
      assign workload_size = data_i[size_width_p-1:0];
    end
    else if (outputs_config_p == 1) begin: split_evenly
      assign workload_size = data_i[size_width_p-1:0] / num_out_p;
    end
    else begin
      $error("outputs_config_p wrong!\n");
    end

    genvar i;
    for (i = 0; i < num_out_p; i=i+1) begin: outputs_add
      assign data_o[i] = {workload_id, workload_size};
    end

  end

endmodule
