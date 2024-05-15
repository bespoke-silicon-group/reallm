include Makefile.pyenv

export MAGIC_NUMBERS_PATH = $(abspath ./vlsi_numbers)
export MICRO_ARCH_PATH = $(abspath ./micro_arch_sim)
export STRUCTS_PATH = $(abspath ./structs)
export PLOT_SCRIPTS_PATH = $(abspath ./plot)
export PYTHONPATH := ${PYTHONPATH}:$(PLOT_SCRIPTS_PATH):$(STRUCTS_PATH):$(MAGIC_NUMBERS_PATH):$(MICRO_ARCH_PATH)

HARDWARE := $(subst .yaml,,$(shell ls ./configs/hardware))
MODELS := $(subst .yaml,,$(shell ls ./configs/models))

CONFIG_DIR = configs
HARDWARE_CONFIG_DIR = $(CONFIG_DIR)/hardware
MODELS_CONFIG_DIR = $(CONFIG_DIR)/models

VERBOSE ?= false

OUTPUT_DIR = outputs
# Create output directory if it does not exist
$(OUTPUT_DIR):
	@mkdir $@ || :

# Hardware Exploration
define HW_GEN
$(hardware).hw: $(OUTPUT_DIR)/$(hardware)/$(hardware).pkl
$(OUTPUT_DIR)/$(hardware)/$(hardware).pkl: $(HARDWARE_CONFIG_DIR)/$(hardware).yaml | $(OUTPUT_DIR) pyenv_exists
	@echo "Running hardware exploration for $(hardware)"
	@if [ "$(VERBOSE)" = "true" ]; then \
		$(VENV_PYTHON3) phases/hardware_exploration.py --config-file $$< --results-dir $(OUTPUT_DIR) --verbose; \
	else \
		$(VENV_PYTHON3) phases/hardware_exploration.py --config-file $$< --results-dir $(OUTPUT_DIR); \
	fi
$(hardware).hw.clean:
	rm -f $(OUTPUT_DIR)/$(hardware)/$(hardware).pkl
	rm -f $(OUTPUT_DIR)/$(hardware)/$(hardware).csv
$(hardware).clean:
	rm -rf $(OUTPUT_DIR)/$(hardware)
endef
$(foreach hardware,$(HARDWARE), \
  $(eval $(call HW_GEN)))

# Software Evaluation
define SW_GEN
$(hardware).sw.$(model): $(OUTPUT_DIR)/$(hardware)/$(model).pkl | pyenv_exists
$(OUTPUT_DIR)/$(hardware)/$(model).pkl: $(MODELS_CONFIG_DIR)/$(model).yaml $(OUTPUT_DIR)/$(hardware)/$(hardware).pkl
	@if [ "$(VERBOSE)" = "true" ]; then \
		$(VENV_PYTHON3) phases/software_evaluation.py --model $$< --hardware $(OUTPUT_DIR)/$(hardware)/$(hardware).pkl --hw-config $(HARDWARE_CONFIG_DIR)/$(hardware).yaml --results-dir $(OUTPUT_DIR) --verbose; \
	else \
		$(VENV_PYTHON3) phases/software_evaluation.py --model $$< --hardware $(OUTPUT_DIR)/$(hardware)/$(hardware).pkl --hw-config $(HARDWARE_CONFIG_DIR)/$(hardware).yaml --results-dir $(OUTPUT_DIR); \
	fi
$(hardware).sw.$(model).clean:
	rm -f $(OUTPUT_DIR)/$(hardware)/$(model).pkl
endef
$(foreach hardware,$(HARDWARE), \
  $(foreach model,$(MODELS), \
	$(eval $(call SW_GEN))))

define SW_ALL_GEN
$(hardware).sw: $(hardware).sw.all
$(hardware).sw.all: $(foreach model,$(MODELS),$(OUTPUT_DIR)/$(hardware)/$(model).pkl)
$(hardware).sw.clean: $(foreach model,$(MODELS),$(hardware).sw.$(model).clean)
endef
$(foreach hardware,$(HARDWARE), \
  $(eval $(call SW_ALL_GEN)))

# Test
test:
	@make test_hw.clean
	$(VENV_PYTHON3) main.py -hw $(HARDWARE_CONFIG_DIR)/test_hw.yaml -m $(MODELS_CONFIG_DIR)/gpt3.yaml
	$(VENV_PYTHON3) tests/compare.py --hardware test_hw --model gpt3

# Clean all
clean:
	rm -rf $(OUTPUT_DIR)
