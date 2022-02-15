// Generate data

`include "bsg_defines.v"

module data_gen #(
   parameter `BSG_INV_PARAM(id_width_p)
  ,parameter `BSG_INV_PARAM(size_width_p)
  ,parameter `BSG_INV_PARAM(num_ports_p)
  ,parameter `BSG_INV_PARAM(els_p)
  ,parameter `BSG_INV_PARAM(workload_limit_p)
  ,parameter `BSG_INV_PARAM(inital_size_p)
  ,parameter `BSG_INV_PARAM(gen_freq_p) // 0: whenever ready; 
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


  logic [num_ports_p-1:0] [31:0] counter;
  always_ff @(posedge clk_i) begin
    if (reset_i) begin
        rom[0] <= inital_size_p / num_ports_p;
        rom[1] <= inital_size_p / num_ports_p;
        flag <= 1;
    end
  end

  for (genvar i = 0; i < num_ports_p; i++) begin
    always_ff @(posedge clk_i) begin
      if (reset_i) begin
        rd_addr[i] <= '0;
        v_o[i] <= '0;
        workload_id[i] <= '0;
        counter[i] <= 0;
      end
      else if (flag & workload_id[i]<workload_limit_p) begin
        if (gen_freq_p == 0) begin
          if (ready_i[i] & v_o[i]) begin
            rd_addr[i] <= rd_addr[i]+1'b1;
            workload_id[i] <= workload_id[i]+1'b1;
          end
          v_o[i] <= '1;
        end
        else begin
          if (counter[i]>=(gen_freq_p-1)) begin
            if (ready_i[i]) begin
              v_o[i] <= '1;
              workload_id[i] <= workload_id[i]+1'b1;
              counter[i] <= 0;
            end
            else begin
              v_o[i] <= '0;
              counter[i] <= counter[i] + 1;
            end
          end
          else begin
            v_o[i] <= '0;
            counter[i] <= counter[i] + 1;
          end
        end
      end
      else begin
        v_o[i] <= 0;
      end
    end
  end

  for (genvar i = 0; i < num_ports_p; i++) begin
    assign data_o[i] = {workload_id[i], rom[rd_addr[i]]};
  end

endmodule

