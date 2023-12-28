export MAGIC_NUMBERS_PATH = $(abspath ../chiplet_cloud_simulator_vlsi_numbers)
export PYTHONPATH := ${PYTHONPATH}:$(MAGIC_NUMBERS_PATH)

# 150 mm2, 40,000 bits/cycle = 5 TB/s --> 352.125 MB
#	python3 ./design_memory.py 150e6 40000 16
# 80 mm2, 22,000 bits/cycle = 2.75 TB/s --> 236.5 MB
#	python3 ./design_memory.py 80e6 22000 16
# 30 mm2, 16,000 bits/cycle = 2 TB/s --> 78.125 MB
#	python3 ./design_memory.py 30e6 16000 16
test:
	python3 ./design_memory.py 30e6 16000 16

# 353 MB, 40,000 bits/cycle = 5 TB/s --> 145.5 mm2
#	python3 ./design_memory_return_area.py 353 40000 16
# 236.5 MB, 22,000 bits/cycle = 2.75 TB/s --> 79 mm2
#	python3 ./design_memory_return_area.py 236.5 22000 16
# 78.125 MB, 16,000 bits/cycle = 2 TB/s --> 28.6 mm2
#	python3 ./design_memory_return_area.py 78.125 16000 16
test2:
	python3 ./design_memory_return_area.py 78.125 16000 16
