// Link

`include "bsg_defines.v"

module link #(
   parameter `BSG_INV_PARAM(id_width_p)
  ,parameter `BSG_INV_PARAM(size_width_p)
  ,parameter data_bytes_p = 2 // bytes / data

  ,parameter `BSG_INV_PARAM(bandwidth_p) // bytes / cycle

  ,parameter width_p = id_width_p + size_width_p
  ,parameter bytes_width_p = size_width_p + `BSG_SAFE_CLOG2(data_bytes_p)
  ,parameter cycles_width_p = bytes_width_p - `BSG_SAFE_CLOG2(bandwidth_p)
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

  logic [width_p-1:0] data_to_ctr, data_from_ctr;
  logic yumi_from_ctr, valid_to_ctr, valid_from_ctr, ready_to_ctr;


  bsg_two_fifo #(.width_p(width_p)) input_fifo
  (  .clk_i   (clk_i)
    ,.reset_i (reset_i)
    // input side
    ,.ready_o (ready_o)
    ,.data_i  (data_i)
    ,.v_i     (v_i)    
    // output side
    ,.v_o     (valid_to_ctr)
    ,.data_o  (data_to_ctr) 
    ,.yumi_i  (yumi_from_ctr)
    );

  wire [id_width_p-1:0] workload_id = data_to_ctr[width_p-1:size_width_p];
  wire [size_width_p-1:0] workload_size = data_to_ctr[size_width_p-1:0];

  wire [bytes_width_p-1:0] data_bytes = workload_size * data_bytes_p;
  // logic [bytes_width_p-1:0] data_bytes;
  // if (data_bytes_p == 2) begin
  //   data_bytes = num_data << 1;
  // end
  // else if (data_bytes_p == 4) begin
  //   data_bytes = num_data << 2;
  // end
  // else begin
  //   $error("Wrong data bytes");
  // end

  wire [cycles_width_p-1:0] cycles = data_bytes / bandwidth_p;
  // wire [cycles_width_p-1:0] cycles = num_data >> $BSG_SAFE_CLOG2(bandwidth_p);

  counter #(
    .cycles_width_p(cycles_width_p)
   ,.width_p       (width_p)
  ) counter_inst
  (  .clk_i    (clk_i)
    ,.reset_i  (reset_i)
    // input side
    ,.yumi_o   (yumi_from_ctr)
    ,.data_i   (data_to_ctr)
    ,.cycles_i (cycles)
    ,.v_i      (valid_to_ctr)    
    // output side
    ,.v_o      (valid_from_ctr)
    ,.data_o   (data_from_ctr) 
    ,.ready_i  (ready_to_ctr)
  );
  
  wire yumi = ready_i & v_o;
  bsg_two_fifo #(.width_p(width_p)) output_fifo
  (  .clk_i   (clk_i)
    ,.reset_i (reset_i)
    // input side
    ,.ready_o (ready_to_ctr)
    ,.data_i  (data_from_ctr)
    ,.v_i     (valid_from_ctr)    
    // output side
    ,.v_o     (v_o)
    ,.data_o  (data_o) 
    ,.yumi_i  (yumi)
    );

endmodule

