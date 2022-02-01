// Generate data

`include "bsg_defines.v"

module data_gen #(
   parameter `BSG_INV_PARAM(id_width_p)
  ,parameter `BSG_INV_PARAM(size_width_p)
  ,parameter `BSG_INV_PARAM(num_ports_p)
  ,parameter `BSG_INV_PARAM(els_p)
  ,parameter `BSG_INV_PARAM(workload_limit_p)
  ,parameter addr_width_lp=`BSG_SAFE_CLOG2(els_p)
  ,parameter width_p = id_width_p + size_width_p
)
  (   input clk_i
    , input reset_i

    , output logic [num_ports_p-1:0]                v_o   
    , output       [num_ports_p-1:0]  [width_p-1:0] data_o
    , input        [num_ports_p-1:0]                ready_i 
    );

  logic [size_width_p-1:0] rom [els_p-1:0];
  logic [num_ports_p-1:0] [addr_width_lp-1:0] rd_addr;

  logic flag;
  logic [num_ports_p-1:0] [id_width_p-1:0] workload_id;


  always_ff @(posedge clk_i) begin
    if (reset_i) begin
        rom[0] <= 1;
        rom[1] <= 1;
        flag <= 1;
    end
  end

  for (genvar i = 0; i < num_ports_p; i++) begin
    always_ff @(posedge clk_i) begin
      if (reset_i) begin
        rd_addr[i] <= '0;
        v_o[i] <= '0;
        workload_id[i] <= '0;
      end
      else if (flag) begin
        if (ready_i[i] & v_o[i]) begin
          rd_addr[i] <= rd_addr[i]+1'b1;
          workload_id[i] <= workload_id[i]+1'b1;
        end
        if (workload_id[i]+1==workload_limit_p) v_o[i] <= 1'b0;
        else v_o[i] <= '1;
      end
    end
  end

  for (genvar i = 0; i < num_ports_p; i++) begin
    assign data_o[i] = {workload_id[i], rom[rd_addr[i]]};
  end

endmodule

