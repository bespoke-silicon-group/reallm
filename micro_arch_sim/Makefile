export MAGIC_NUMBERS_PATH = $(abspath ../chiplet_cloud_simulator_vlsi_numbers)
export PLOT_SCRIPTS_PATH = $(abspath ../plot)
export ASIC_CLOUD_SCRIPTS_PATH = $(abspath ../hardware_exploration/scripts)
export PYTHONPATH := ${PYTHONPATH}:$(PLOT_SCRIPTS_PATH):$(ASIC_CLOUD_SCRIPTS_PATH):$(MAGIC_NUMBERS_PATH)

test:
	python3 ./design_memory.py 300000000 65536 16
