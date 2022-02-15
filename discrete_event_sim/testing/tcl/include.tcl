
set basejump_stl_dir       $::env(BASEJUMP_STL_DIR)
set bsg_designs_target_dir $::env(BSG_DESIGNS_TARGET_DIR)

# set bsg_pinout        $::env(BSG_PINOUT)
# set bsg_padmapping    $::env(BSG_PADMAPPING)

set TESTING_INCLUDE_PATHS [join "
  $bsg_designs_target_dir/v
  $basejump_stl_dir/bsg_misc
  $basejump_stl_dir/bsg_cache
  $basejump_stl_dir/bsg_clk_gen
  $basejump_stl_dir/bsg_noc
  $basejump_stl_dir/bsg_tag
"]
# $bsg_packaging_dir/$bsg_package/pinouts/$bsg_pinout/common/verilog
