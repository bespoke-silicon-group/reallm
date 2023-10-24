export MICRO_ARCH_PATH = $(abspath ./micro_arch_sim)
export HW_STRUCTS_PATH = $(abspath ./structs)
export PYTHONPATH := ${PYTHONPATH}:$(MICRO_ARCH_PATH)

CONFIG = 'exploration'

all: gen

gen:
	python chiplet_system_gen.py --config $(CONFIG)

test:
	python chiplet_system_gen.py --config 'test'

clean:
	-rm -f *.tco
	-rm -f *.csv
	-rm -f *.pdf
	-rm -f *.plt
	-rm -f *.d
	-rm -f *.svg
	-rm -f *.html

