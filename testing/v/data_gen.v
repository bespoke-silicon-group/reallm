// Generate data

`include "bsg_defines.v"

module data_gen #(
   parameter `BSG_INV_PARAM(id_width_p)
  ,parameter `BSG_INV_PARAM(size_width_p)
  ,parameter `BSG_INV_PARAM(els_p)
  ,parameter `BSG_INV_PARAM(workload_limit_p)
  ,parameter addr_width_lp=`BSG_SAFE_CLOG2(els_p)
  ,parameter width_p = id_width_p + size_width_p
)
  ( input clk_i
    , input reset_i

    , output logic        v_o   
    , output[width_p-1:0] data_o
    , input               ready_i 
    );

  logic [size_width_p-1:0] rom [els_p-1:0];
  logic [addr_width_lp-1:0] rd_addr;

  logic flag;
  logic [id_width_p-1:0] workload_id;

  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      rd_addr <= 0;
      v_o <= 1'b0;
      workload_id <= 0;

      rom[0] <= 2;
      rom[1] <= 2;

      flag <= 1;
    end
    else if (flag) begin
      if (ready_i & v_o) begin
        $display("Send workload %d at time %d", workload_id, $time);
        rd_addr <= rd_addr+1'b1;
        workload_id <= workload_id+1'b1;
      end
      if (workload_id+1==workload_limit_p) v_o <= 1'b0;
      else v_o <= 1'b1;
    end
  end

  assign data_o = {workload_id, rom[rd_addr]};


endmodule

