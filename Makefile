include Makefile.setup

# Set base paths
GIT_ROOT := $(shell git rev-parse --show-toplevel)
VENV_PYTHON3 := source $(GIT_ROOT)/pyenv/bin/activate && python3


# Clean env and all generated artifacts
.PHONY: clean_all
clean_all:
	$(MAKE) -f Makefile.setup clean_pyenv


