export PYTHON = ~/miniconda3/envs/py2/bin/python
export SCRIPTS_PATH = $(abspath ../asic_cloud_sim/scripts)
export PYTHONPATH := ${PYTHONPATH}:$(SCRIPTS_PATH)

IMAGE_FORMAT ?= pdf
export IMAGE_FORMAT

gen:
	python gpt_analysis.py

plot:
	$(PYTHON) plot.py

clean:
	-rm -f *.csv
	-rm -f *.pdf

