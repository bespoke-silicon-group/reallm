export LD_LIBRARY_PATH := /mnt/users/ssd3/homes/huwan/tool/gcc-12.3.0/lib64:${LD_LIBRARY_PATH}

# Please set the path of DRAMsim3 executable file here
DRAMSIM3 = ../../DRAMsim3/build/dramsim3main

CONFIG = dram_configs/HBM2_8GB/2_stack.ini
TRACE = dramsim3_stream_i1_n2000000_rw2000000_c8.trace
CYCLE = 1000

run:
	python3 run_dramsim3.py

single_run:
	$(DRAMSIM3) $(CONFIG) -t $(TRACE) -c $(CYCLE) --host=a