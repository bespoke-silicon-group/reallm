`include "bsg_defines.v"

module chiplets_1x2_tb;

  parameter period_p = 10; // pico second

  parameter id_width_p = 8;
  parameter size_width_p = 8;
  parameter data_bytes_p = 2;

  parameter link_bandwidth_p = 4; // bytes / cycle
  parameter num_macs_p = 4; // macs / cycle

  localparam width_p = id_width_p + size_width_p;


  logic clk_i;
  logic reset_i;

  logic ready_o;
  logic [width_p-1:0] data_i;
  logic v_i;

  logic v_o;
  logic [width_p-1:0] data_o;
  logic ready_i;


  bsg_nonsynth_clock_gen #(.cycle_time_p(period_p))    sim_clk     (.o(clk_i));

  initial $display("%m creating clocks", period_p);

  bsg_nonsynth_reset_gen
    #(.num_clocks_p(1)
      ,.reset_cycles_lo_p(6)
      ,.reset_cycles_hi_p(4)
      ) reset_gen
      (.clk_i(clk_i)
       ,.async_reset_o(reset_i)
       );

  data_gen #(
     .id_width_p(id_width_p)
    ,.size_width_p(size_width_p)
    ,.els_p(2)
    ,.workload_limit_p(16)
  ) data_gen_inst
      ( .clk_i(clk_i)
      , .reset_i(reset_i)

      , .v_o(v_i)
      , .data_o(data_i) 
      , .ready_i(ready_o)
      );

  chiplets_1x2 #(
     .id_width_p(id_width_p)
    ,.size_width_p(size_width_p)
    ,.data_bytes_p(data_bytes_p)
    ,.num_macs_p(num_macs_p)
    ,.bandwidth_p(link_bandwidth_p)
  ) inst
      ( .clk_i(clk_i)
      , .reset_i(reset_i)
  
      // input side
      , .ready_o(ready_o)
      , .data_i(data_i)
      , .v_i(v_i)    
  
      // output side
      , .v_o(v_o)
      , .data_o(data_o) 
      , .ready_i(ready_i)
      );

  assign ready_i = 1;
  initial begin
    $printtimescale;
    #5000
    $finish;
  end

  always_ff @(posedge clk_i) begin
    if (v_o) begin
      $display("Received workload %d at time %d", data_o[width_p-1 -: id_width_p], $time);
    end
  end

endmodule

