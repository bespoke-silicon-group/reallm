#------------------------------------------------------------
# Do NOT arbitrarily change the order of files. Some module
# and macro definitions may be needed by the subsequent files
#------------------------------------------------------------

set basejump_stl_dir       $::env(TESTING_BASEJUMP_STL_DIR)
set bsg_designs_dir        $::env(TESTING_BSG_DESIGNS_DIR)
set bsg_designs_target_dir $::env(TESTING_BSG_DESIGNS_TARGET_DIR)

set TESTING_PACKAGE_FILES [join "
  $basejump_stl_dir/bsg_misc/bsg_defines.v
"]

set TESTING_SOURCE_FILES [join "
  $TESTING_PACKAGE_FILES

  $bsg_designs_target_dir/testing/v/chiplets_1x2_tb.v
  $bsg_designs_target_dir/testing/v/data_gen.v

  $bsg_designs_target_dir/v/chiplets_1x2.v
  $bsg_designs_target_dir/v/chiplet.v
  $bsg_designs_target_dir/v/link.v
  $bsg_designs_target_dir/v/counter.v
  
  $basejump_stl_dir/bsg_misc/bsg_dff.v
  $basejump_stl_dir/bsg_misc/bsg_dff_reset.v
  $basejump_stl_dir/bsg_dataflow/bsg_two_fifo.v
  $basejump_stl_dir/bsg_test/bsg_nonsynth_clock_gen.v
  $basejump_stl_dir/bsg_test/bsg_nonsynth_reset_gen.v

  $basejump_stl_dir/bsg_mem/bsg_mem_1r1w_synth.v
  $basejump_stl_dir/bsg_mem/bsg_mem_1r1w.v
"]

