// Counter

`include "bsg_defines.v"

module counter #(
   parameter `BSG_INV_PARAM(cycles_width_p)
  ,parameter `BSG_INV_PARAM(width_p)
)
  ( input clk_i
    , input reset_i

    // input side
    , output                     yumi_o 
    , input [cycles_width_p-1:0] cycles_i 
    , input [width_p-1:0]        data_i
    , input                      v_i     

    // output side
    , output                     v_o   
    , output [width_p-1:0]       data_o
    , input                      ready_i 
    );


  logic [cycles_width_p-1:0] counter_num;
  logic idle, finished;

  logic [width_p-1:0] data_reg;

  assign data_o = data_reg;

  always_ff @(posedge clk_i) begin
    if (reset_i) begin
      counter_num <= cycles_width_p'(0);
      idle <= 1'b1;
      finished <= 1'b0;
    end
    else if (idle) begin
      if (v_i) begin
        counter_num <= cycles_i;
        idle <= 1'b0;
        finished <= 1'b0;
        data_reg <= data_i;
      end
      else begin
        counter_num <= cycles_width_p'(0);
        idle <= 1'b1;
        finished <= 1'b0;
      end
    end
    else if (~finished) begin
      counter_num <= counter_num - 1'b1;
      if (counter_num==1'b1) begin
        finished <= ready_i;
        idle <= ready_i;
      end
      else begin
        finished <= 1'b0;
        idle <= 1'b0;
      end
    end
    else begin
      finished <= ~ready_i;
      idle <= ready_i;
    end
  end

  assign v_o = finished & ready_i;
  assign yumi_o = v_i & idle; 

endmodule

