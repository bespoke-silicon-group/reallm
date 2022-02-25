export PYTHON = ~/miniconda3/envs/py2/bin/python
export SCRIPTS_PATH = $(abspath ../asic_cloud_sim/scripts)
export PYTHONPATH := ${PYTHONPATH}:$(SCRIPTS_PATH)

IMAGE_FORMAT ?= pdf
export IMAGE_FORMAT

gen:
	python gen_opt_routing.py

best:
	python best_system.py

plot_pareto:
	$(PYTHON) plot_pareto.py

plot_system:
	python plot_system.py

clean:
	-rm -f *.csv
	-rm -f *.pdf
	-rm -f *.gv

