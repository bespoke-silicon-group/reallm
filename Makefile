include Makefile.setup

# Use bash as the shell
SHELL := /bin/bash

# Set base paths
GIT_ROOT := $(shell git rev-parse --show-toplevel)

# Python virtual env
VENV_PYTHON := source $(GIT_ROOT)/pyenv/bin/activate && python3

# Default parameters
WORKSPACE_DIR := workspace
CONFIG_STORE := $(WORKSPACE_DIR)/.current_config
CONFIG_FILE := $(if $(wildcard $(CONFIG_STORE)),$(shell cat $(CONFIG_STORE)),configs/system/default_homo.yaml)

# Simulation parameters
SIM_TRACE :=
SIM_TASK :=
SIM_RATE :=

# Main targets
.PHONY: all kernel traces sim clean run-config clean_all clean_workspace set_config current_config

# Set configuration file
set_config:
	@if [ -z "$(CONFIG)" ]; then \
		echo "Please specify CONFIG=<path-to-config>"; \
		exit 1; \
	fi; \
	mkdir -p $(WORKSPACE_DIR); \
	echo $(CONFIG) > $(CONFIG_STORE); \
	echo "Configuration set to: $(CONFIG)"

# Display current configuration
current_config:
	@if [ -f $(CONFIG_STORE) ]; then \
		echo "Current configuration: $(shell cat $(CONFIG_STORE))"; \
	else \
		echo "Using default configuration: configs/system/default_homo.yaml"; \
	fi

# Run all steps
all:
	$(VENV_PYTHON) main.py --config $(CONFIG_FILE) --workspace_dir $(WORKSPACE_DIR) --mode all

# Generate kernel sizes and run kernel simulation
kernel:
	$(VENV_PYTHON) main.py --config $(CONFIG_FILE) --workspace_dir $(WORKSPACE_DIR) --mode kernel

# Download and generate traces
traces:
	$(VENV_PYTHON) main.py --config $(CONFIG_FILE) --workspace_dir $(WORKSPACE_DIR) --mode trace

# Run system simulation
sim:
	$(VENV_PYTHON) main.py --config $(CONFIG_FILE) --workspace_dir $(WORKSPACE_DIR) --mode sim \
	$(if $(TRACE),--trace $(TRACE),) \
	$(if $(TASK),--task $(TASK),) \
	$(if $(RATE),--rate $(RATE),)

# Run with a specific configuration
run-config:
	@if [ -z "$(CONFIG)" ]; then \
		echo "Please specify CONFIG=<path-to-config>"; \
		exit 1; \
	fi; \
	$(VENV_PYTHON) main.py --config $(CONFIG) --workspace_dir $(WORKSPACE_DIR) --mode all

# Clean all workspace (all subdirectories)
clean_workspace:
	@read -p "Do you really want to clean the entire workspace? [y/N] " confirm; \
	if [[ $$confirm == [yY] || $$confirm == [yY][eE][sS] ]]; then \
		echo "Cleaning entire workspace..."; \
		rm -rf $(WORKSPACE_DIR)/kernel_lib/*; \
		rm -rf $(WORKSPACE_DIR)/traces/*; \
		rm -rf $(WORKSPACE_DIR)/llmcompass_results/*; \
		rm -rf $(WORKSPACE_DIR)/roofline_results/*; \
		echo "All workspace directories cleaned."; \
	else \
		echo "Clean operation canceled."; \
	fi

# Clean env and all generated artifacts
clean_all:
	@read -p "Do you really want to clean workspace AND virtual environment? This will require re-running setup. [y/N] " confirm; \
	if [[ $$confirm == [yY] || $$confirm == [yY][eE][sS] ]]; then \
		echo "Cleaning workspace and virtual environment..."; \
		$(MAKE) clean_workspace; \
		$(MAKE) -f Makefile.setup clean_pyenv; \
		echo "Workspace and virtual environment cleaned."; \
	else \
		echo "Clean operation canceled."; \
	fi


