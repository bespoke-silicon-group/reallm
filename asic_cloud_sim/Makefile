export PYTHON = ~/miniconda3/envs/py2/bin/python
export SCRIPTS_PATH = $(abspath ./scripts)
export PYTHONPATH := ${PYTHONPATH}:$(SCRIPTS_PATH)

IMAGE_FORMAT ?= pdf
export IMAGE_FORMAT

all: gen opt

gen:
	$(PYTHON) chiplet_system_gen.py

opt:
	$(PYTHON) chiplet_system_opt.py

plot:
	$(PYTHON) chiplet_system_plot.py

clean:
	-rm -f *.tco
	-rm -f *.csv
	-rm -f *.pdf
	-rm -f *.plt
	-rm -f *.d
	-rm -f *.svg
	-rm -f *.html

