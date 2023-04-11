export SCRIPTS_PATH = $(abspath ./scripts)
export PYTHONPATH := ${PYTHONPATH}:$(SCRIPTS_PATH)

CONFIG = 'HBM_chiplet'

all: gen

gen:
	python chiplet_system_gen.py --config $(CONFIG)

clean:
	-rm -f *.tco
	-rm -f *.csv
	-rm -f *.pdf
	-rm -f *.plt
	-rm -f *.d
	-rm -f *.svg
	-rm -f *.html

