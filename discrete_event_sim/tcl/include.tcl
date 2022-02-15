
set basejump_stl_dir       $::env(BASEJUMP_STL_DIR)
set bsg_designs_target_dir $::env(BSG_DESIGNS_TARGET_DIR)

# set SVERILOG_INCLUDE_PATHS [join "
#   $bsg_packaging_dir/common/verilog
#   $bsg_packaging_dir/common/foundry/portable/verilog
#   $bsg_packaging_dir/$bsg_package/pinouts/$bsg_pinout/common/verilog
#   $bsg_packaging_dir/$bsg_package/pinouts/$bsg_pinout/portable/verilog
#   $bsg_packaging_dir/$bsg_package/pinouts/$bsg_pinout/portable/verilog/padmappings/$bsg_padmapping
#   $bsg_designs_target_dir/v
#   $bsg_designs_target_dir/HardFloat/source/
#   $bsg_designs_target_dir/HardFloat/source/RISCV
#   $basejump_stl_dir/bsg_misc
#   $basejump_stl_dir/bsg_cache
#   $basejump_stl_dir/bsg_clk_gen
#   $basejump_stl_dir/bsg_noc
#   $basejump_stl_dir/bsg_tag
# "]

set SVERILOG_INCLUDE_PATHS [join "
  $bsg_designs_target_dir/v
  $basejump_stl_dir/bsg_misc
  $basejump_stl_dir/bsg_cache
  $basejump_stl_dir/bsg_clk_gen
  $basejump_stl_dir/bsg_noc
  $basejump_stl_dir/bsg_tag
"]

