// Chiplets Array

`include "bsg_defines.v"

module chiplets_array #(
   parameter `BSG_INV_PARAM(id_width_p)
  ,parameter `BSG_INV_PARAM(size_width_p)
  ,parameter data_bytes_p = 2 // bytes / data

  // chiplets array size
  ,parameter `BSG_INV_PARAM(num_chiplets_x_p)
  ,parameter `BSG_INV_PARAM(num_chiplets_y_p)

  // chiplets array routing shemes
  ,parameter `BSG_INV_PARAM(chiplets_routing_p) // Type 0 or Type 1

  // chiplets hardware settings
  ,parameter `BSG_INV_PARAM(num_macs_p) // macs / cycle
  ,parameter `BSG_INV_PARAM(bandwidth_p) // bytes / cycle

  // chiplets configuration
  ,parameter integer macs_per_data_p[num_chiplets_x_p-1:0]

  ,parameter width_p = id_width_p + size_width_p
)
  (   input clk_i
    , input reset_i

    // input side
    , output [num_chiplets_y_p-1:0]               ready_o 
    , input  [num_chiplets_y_p-1:0] [width_p-1:0] data_i
    , input  [num_chiplets_y_p-1:0]               v_i     

    // output side
    , output [num_chiplets_y_p-1:0]               v_o   
    , output [num_chiplets_y_p-1:0] [width_p-1:0] data_o
    , input  [num_chiplets_y_p-1:0]               ready_i 
    );

  if (num_chiplets_y_p == 1) begin: one_row
    // horizontal links
    logic [num_chiplets_x_p:0] hor_ready_o, hor_ready_i;
    logic [num_chiplets_x_p:0][width_p-1:0] hor_data_i, hor_data_o;
    logic [num_chiplets_x_p:0] hor_valid_i, hor_valid_o;

    assign ready_o = hor_ready_i[0];
    assign hor_data_o[0] = data_i;
    assign hor_valid_o[0] = v_i;

    for (genvar c = 0; c < num_chiplets_x_p; c++) begin: x
        chiplet #(
           .id_width_p(id_width_p)
          ,.size_width_p(size_width_p)
          ,.data_bytes_p(data_bytes_p)
          ,.num_macs_p(num_macs_p)
          ,.num_in_p(1)
          ,.num_out_p(1)
          ,.inputs_select_p(0)
          ,.outputs_config_p(0)
          ,.macs_per_data_p(macs_per_data_p[c])
        ) chip
            ( .clk_i(clk_i)
            , .reset_i(reset_i)
            // input side
            , .ready_o(hor_ready_i[c])
            , .data_i(hor_data_o[c])
            , .v_i(hor_valid_o[c])    
            // output side
            , .v_o(hor_valid_i[c+1])
            , .data_o(hor_data_i[c+1]) 
            , .ready_i(hor_ready_o[c+1])
            );
      if (c < num_chiplets_x_p - 1) begin
        link #(
           .id_width_p(id_width_p)
          ,.size_width_p(size_width_p)
          ,.data_bytes_p(data_bytes_p)
          ,.bandwidth_p(bandwidth_p)
        ) link
            ( .clk_i(clk_i)
            , .reset_i(reset_i)
            // input side
            , .ready_o(hor_ready_o[c+1])
            , .data_i(hor_data_i[c+1])
            , .v_i(hor_valid_i[c+1])    
            // output side
            , .v_o(hor_valid_o[c+1])
            , .data_o(hor_data_o[c+1]) 
            , .ready_i(hor_ready_i[c+1])
            );
      end
      else begin
        assign v_o = hor_valid_i[c+1];
        assign data_o = hor_data_i[c+1];
        assign hor_ready_o[c+1] = ready_i;
      end
    end

    // Simulation log for single row chipltes
    for (genvar i = 0; i < num_chiplets_x_p+1; i++) begin
      always_ff @(posedge clk_i) begin
        // chiplet valid: hor_valid_i
        if (hor_valid_i[i] && hor_ready_o[i]) begin
          $fwrite(f, "chip ( 0,%2d)          finished %d at cycle %d\n", i-1, hor_data_i[i][width_p-1 -: id_width_p], $time/10);
        end
        // link valid: hor_valid_o
        if (hor_valid_o[i+1] && hor_ready_i[i+1]) begin
          $fwrite(f, "link ( 0,%2d)-->( 0,%2d) finished %d at cycle %d\n", i, i+1, hor_data_o[i+1][width_p-1 -: id_width_p], $time/10);
        end
      end
    end

  end
  else if (chiplets_routing_p == 0) begin: type0_routing
    // horizontal links
    logic [num_chiplets_x_p:0][num_chiplets_y_p-1:0] hor_ready_o, hor_ready_i;
    logic [num_chiplets_x_p:0][num_chiplets_y_p-1:0][width_p-1:0] hor_data_i, hor_data_o;
    logic [num_chiplets_x_p:0][num_chiplets_y_p-1:0] hor_valid_i, hor_valid_o;
    // vertical links
    logic [num_chiplets_x_p-1:0][num_chiplets_y_p-2:0] ver_ready_o, ver_ready_i;
    logic [num_chiplets_x_p-1:0][num_chiplets_y_p-2:0][width_p-1:0] ver_data_i, ver_data_o;
    logic [num_chiplets_x_p-1:0][num_chiplets_y_p-2:0] ver_valid_i, ver_valid_o;

    assign ready_o = hor_ready_i[0];
    assign hor_data_o[0] = data_i;
    assign hor_valid_o[0] = v_i;

    for (genvar c = 0; c < num_chiplets_x_p; c++) begin: x
      for (genvar r = 0; r < num_chiplets_y_p; r++) begin: y
        if (r == 0) begin
          chiplet #(
             .id_width_p(id_width_p)
            ,.size_width_p(size_width_p)
            ,.data_bytes_p(data_bytes_p)
            ,.num_macs_p(num_macs_p)
            ,.num_in_p(1)
            ,.num_out_p(1)
            ,.inputs_select_p(0)
            ,.outputs_config_p(0)
            ,.macs_per_data_p(macs_per_data_p[c])
          ) chip
              ( .clk_i(clk_i)
              , .reset_i(reset_i)
              // input side
              , .ready_o(hor_ready_i[c][r])
              , .data_i(hor_data_o[c][r])
              , .v_i(hor_valid_o[c][r])    
              // output side
              , .v_o(ver_valid_i[c][r])
              , .data_o(ver_data_i[c][r]) 
              , .ready_i(ver_ready_o[c][r])
              );
          link #(
             .id_width_p(id_width_p)
            ,.size_width_p(size_width_p)
            ,.data_bytes_p(data_bytes_p)
            ,.bandwidth_p(bandwidth_p)
          ) link
              ( .clk_i(clk_i)
              , .reset_i(reset_i)
              // input side
              , .ready_o(ver_ready_o[c][r])
              , .data_i(ver_data_i[c][r])
              , .v_i(ver_valid_i[c][r])    
              // output side
              , .v_o(ver_valid_o[c][r])
              , .data_o(ver_data_o[c][r]) 
              , .ready_i(ver_ready_i[c][r])
              );

          // Simulation log for type 0 routing chipltes
          always_ff @(posedge clk_i) begin
            // chiplet valid: ver_valid_i
            if (ver_valid_i[c][r] && ver_ready_o[c][r]) begin
              $fwrite(f, "chip (%2d,%2d)           finished %d at cycle %d\n", r, c, ver_data_i[c][r][width_p-1 -: id_width_p], $time/10);
            end
            // link valid: ver_valid_o
            if (ver_valid_o[c][r] && ver_ready_i[c][r]) begin
              $fwrite(f, "link (%2d,%2d)-->(%2d,%2d) finished %d at cycle %d\n", r, c, r+1, c, ver_data_o[c][r][width_p-1 -: id_width_p], $time/10);
            end
          end
        end
        else if (r < num_chiplets_y_p - 1) begin
          // inputs from left and top
          wire [1:0] chiplet_ready_o;
          wire [1:0][width_p-1:0] chiplet_data_i = {ver_data_o[c][r-1], hor_data_o[c][r]};
          wire [1:0] chiplet_v_i = {ver_valid_o[c][r-1], hor_valid_o[c][r]};
          assign ver_ready_i[c][r-1] = chiplet_ready_o[1];
          assign hor_ready_i[c][r] = chiplet_ready_o[0];
          chiplet #(
             .id_width_p(id_width_p)
            ,.size_width_p(size_width_p)
            ,.data_bytes_p(data_bytes_p)
            ,.num_macs_p(num_macs_p)
            ,.num_in_p(2)
            ,.num_out_p(1)
            ,.inputs_select_p(0)
            ,.outputs_config_p(0)
            ,.macs_per_data_p(macs_per_data_p[c])
          ) chip
              ( .clk_i(clk_i)
              , .reset_i(reset_i)
              // input side
              , .ready_o(chiplet_ready_o)
              , .data_i(chiplet_data_i)
              , .v_i(chiplet_v_i)    
              // output side
              , .v_o(ver_valid_i[c][r])
              , .data_o(ver_data_i[c][r]) 
              , .ready_i(ver_ready_o[c][r])
              );
          link #(
             .id_width_p(id_width_p)
            ,.size_width_p(size_width_p)
            ,.data_bytes_p(data_bytes_p)
            ,.bandwidth_p(bandwidth_p)
          ) link
              ( .clk_i(clk_i)
              , .reset_i(reset_i)
              // input side
              , .ready_o(ver_ready_o[c][r])
              , .data_i(ver_data_i[c][r])
              , .v_i(ver_valid_i[c][r])    
              // output side
              , .v_o(ver_valid_o[c][r])
              , .data_o(ver_data_o[c][r]) 
              , .ready_i(ver_ready_i[c][r])
              );
          // Simulation log for type 0 routing chipltes
          always_ff @(posedge clk_i) begin
            // chiplet valid: ver_valid_i
            if (ver_valid_i[c][r] && ver_ready_o[c][r]) begin
              $fwrite(f, "chip (%2d,%2d)            finished %d at cycle %d\n", r, c, ver_data_i[c][r][width_p-1 -: id_width_p], $time/10);
            end
            // link valid: ver_valid_o
            if (ver_valid_o[c][r] && ver_ready_i[c][r]) begin
              $fwrite(f, "link (%2d,%2d)-->(%2d,%2d) finished %d at cycle %d\n", r, c, r+1, c, ver_data_o[c][r][width_p-1 -: id_width_p], $time/10);
            end
          end
        end
        else begin
          // inputs from left and top, outputs to all chips at the next column
          wire [1:0] chiplet_ready_o;
          wire [1:0][width_p-1:0] chiplet_data_i = {ver_data_o[c][r-1], hor_data_o[c][r]};
          wire [1:0] chiplet_v_i = {ver_valid_o[c][r-1], hor_valid_o[c][r]};
          assign ver_ready_i[c][r-1] = chiplet_ready_o[1];
          assign hor_ready_i[c][r] = chiplet_ready_o[0];
          chiplet #(
             .id_width_p(id_width_p)
            ,.size_width_p(size_width_p)
            ,.data_bytes_p(data_bytes_p)
            ,.num_macs_p(num_macs_p)
            ,.num_in_p(2)
            ,.num_out_p(num_chiplets_y_p)
            ,.inputs_select_p(0)
            ,.outputs_config_p(1)
            ,.macs_per_data_p(macs_per_data_p[c])
          ) chip
              ( .clk_i(clk_i)
              , .reset_i(reset_i)
              // input side
              , .ready_o(chiplet_ready_o)
              , .data_i(chiplet_data_i)
              , .v_i(chiplet_v_i)    
              // output side
              , .v_o(hor_valid_i[c+1])
              , .data_o(hor_data_i[c+1]) 
              , .ready_i(hor_ready_o[c+1])
              );
          if (c < num_chiplets_x_p-1) begin
            for (genvar i = 0; i < num_chiplets_y_p; i++) begin
              link #(
                 .id_width_p(id_width_p)
                ,.size_width_p(size_width_p)
                ,.data_bytes_p(data_bytes_p)
                ,.bandwidth_p(bandwidth_p)
              ) link
                  ( .clk_i(clk_i)
                  , .reset_i(reset_i)
                  // input side
                  , .ready_o(hor_ready_o[c+1][i])
                  , .data_i(hor_data_i[c+1][i])
                  , .v_i(hor_valid_i[c+1][i])    
                  // output side
                  , .v_o(hor_valid_o[c+1][i])
                  , .data_o(hor_data_o[c+1][i]) 
                  , .ready_i(hor_ready_i[c+1][i])
                  );
            end
          end
          else begin
            assign v_o = hor_valid_i[c+1];
            assign data_o = hor_data_i[c+1];
            assign hor_ready_o[c+1] = ready_i;
          end
          // Simulation log for type 0 routing chipltes
          always_ff @(posedge clk_i) begin
            // chiplet valid: hor_valid_i
            if (hor_valid_i[c+1][0] && hor_ready_o[c+1][0]) begin
              $fwrite(f, "chip (%2d,%2d)           finished %d at cycle %d\n", r, c, hor_data_i[c+1][0][width_p-1 -: id_width_p], $time/10);
            end
          end
          if (c < num_chiplets_x_p-1) begin
            for (genvar i = 0; i < num_chiplets_y_p; i++) begin
              always_ff @(posedge clk_i) begin
                // link valid: hor_valid_o
                if (hor_valid_o[c+1][i] && hor_ready_i[c+1][i]) begin
                  $fwrite(f, "link (%2d,%2d)-->(%2d,%2d) finished %d at cycle %d\n", r, c, i, c+1, hor_data_o[c+1][i][width_p-1 -: id_width_p], $time/10);
                end
              end
            end
          end
        end
      end
    end
  end
  else if (chiplets_routing_p == 1) begin: type1_routing
    $error("chiplets_routing_p=1 is not implemented!");
  end
  else begin
    $error("chiplets_routing_p is wrong!");
  end


  integer f;
  initial begin
    f = $fopen("sim_detail.log","w");
  end


endmodule

