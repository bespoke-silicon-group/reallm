export SCRIPTS_PATH = $(abspath ./scripts)
export PYTHONPATH := ${PYTHONPATH}:$(SCRIPTS_PATH)

IMAGE_FORMAT ?= pdf
export IMAGE_FORMAT

CONFIG = 'HBM'

all: gen opt plot

gen:
	python chiplet_system_gen.py --config $(CONFIG)

opt:
	python chiplet_system_opt.py --config $(CONFIG)

plot:
	python chiplet_system_plot.py --config $(CONFIG)

clean:
	-rm -f *.tco
	-rm -f *.csv
	-rm -f *.pdf
	-rm -f *.plt
	-rm -f *.d
	-rm -f *.svg
	-rm -f *.html

