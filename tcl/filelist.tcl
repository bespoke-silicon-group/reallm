#------------------------------------------------------------
# Do NOT arbitrarily change the order of files. Some module
# and macro definitions may be needed by the subsequent files
#------------------------------------------------------------

set basejump_stl_dir       $::env(BASEJUMP_STL_DIR)
set bsg_designs_target_dir $::env(BSG_DESIGNS_TARGET_DIR)

set SVERILOG_PACKAGE_FILES [join "
  $basejump_stl_dir/bsg_misc/bsg_defines.v
"]

set SVERILOG_SOURCE_FILES [join "
  $SVERILOG_PACKAGE_FILES

  $bsg_designs_target_dir/v/chiplets_1x2.v
  $bsg_designs_target_dir/v/chiplet.v
  $bsg_designs_target_dir/v/link.v
  $bsg_designs_target_dir/v/counter.v
  $bsg_designs_target_dir/v/inputs_gather.v
  $bsg_designs_target_dir/v/outputs_scatter.v
  $bsg_designs_target_dir/v/inputs_cycles_calculate.v
  $bsg_designs_target_dir/v/outputs_workload_calculate.v

  $basejump_stl_dir/bsg_misc/bsg_dff.v
  $basejump_stl_dir/bsg_misc/bsg_reduce.v
  $basejump_stl_dir/bsg_misc/bsg_dff_reset.v
  $basejump_stl_dir/bsg_dataflow/bsg_two_fifo.v

  $basejump_stl_dir/bsg_mem/bsg_mem_1r1w_synth.v
  $basejump_stl_dir/bsg_mem/bsg_mem_1r1w.v
"]
