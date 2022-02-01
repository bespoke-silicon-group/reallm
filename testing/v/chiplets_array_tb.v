`include "bsg_defines.v"

module chiplets_array_tb;

  parameter period_p = 10; // pico second

  parameter id_width_p = 8;
  parameter size_width_p = 8;
  parameter data_bytes_p = 2;

  parameter num_chiplets_x_p = 2;
  parameter num_chiplets_y_p = 2;
  parameter chiplets_routing_p = 0;

  parameter link_bandwidth_p = 4; // bytes / cycle
  parameter num_macs_p = 4; // macs / cycle

  parameter integer macs_per_data_p[num_chiplets_x_p-1:0] = {2, 4};

  localparam width_p = id_width_p + size_width_p;


  logic clk_i;
  logic reset_i;

  // inputs 
  logic [num_chiplets_y_p-1:0] ready_o;
  logic [num_chiplets_y_p-1:0] [width_p-1:0] data_i;
  logic [num_chiplets_y_p-1:0] v_i;
  // outputs
  logic [num_chiplets_y_p-1:0] v_o;
  logic [num_chiplets_y_p-1:0] [width_p-1:0] data_o;
  logic [num_chiplets_y_p-1:0] ready_i;


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
    ,.num_ports_p(num_chiplets_y_p)
    ,.els_p(2)
    ,.workload_limit_p(30)
  ) data_gen_inst
      ( .clk_i(clk_i)
      , .reset_i(reset_i)

      , .v_o(v_i)
      , .data_o(data_i) 
      , .ready_i(ready_o)
      );

  chiplets_array #(
     .id_width_p(id_width_p)
    ,.size_width_p(size_width_p)
    ,.data_bytes_p(data_bytes_p)
    ,.num_chiplets_x_p(num_chiplets_x_p)
    ,.num_chiplets_y_p(num_chiplets_y_p)
    ,.chiplets_routing_p(chiplets_routing_p)
    ,.num_macs_p(num_macs_p)
    ,.bandwidth_p(link_bandwidth_p)
    ,.macs_per_data_p(macs_per_data_p)
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

  assign ready_i = '1;

  integer f;
  initial begin
    $printtimescale;
    f = $fopen("output.log","w");
    #10000
    $finish;
  end

  for (genvar i = 0; i < num_chiplets_y_p; i++) begin
    always_ff @(posedge clk_i) begin
      if (ready_o[i] & v_i[i]) begin
        $fwrite(f, "Send workload %d to chiplet %2d at cycle %d\n", data_i[i][width_p-1 -: id_width_p], i, $time/10);
      end
    end
  end

  always_ff @(posedge clk_i) begin
    if (v_o[0]) begin
      $fwrite(f, "Received workload %d at cycle %d\n", data_o[0][width_p-1 -: id_width_p], $time/10);
    end
  end

endmodule

