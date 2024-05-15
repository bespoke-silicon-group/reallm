include Makefile.pyenv

export MAGIC_NUMBERS_PATH = $(abspath ./vlsi_numbers)
export MICRO_ARCH_PATH = $(abspath ./micro_arch_sim)
export STRUCTS_PATH = $(abspath ./structs)
export PLOT_SCRIPTS_PATH = $(abspath ./plot)
export PYTHONPATH := ${PYTHONPATH}:$(PLOT_SCRIPTS_PATH):$(STRUCTS_PATH):$(MAGIC_NUMBERS_PATH):$(MICRO_ARCH_PATH)

HARDWARE := $(subst .yaml,,$(shell ls ./inputs/hardware/config))
MODELS := $(subst .yaml,,$(shell ls ./inputs/software/model))

INPUT_DIR = inputs
HARDWARE_INPUT_DIR = $(INPUT_DIR)/hardware/config
MODELS_INPUT_DIR = $(INPUT_DIR)/software/model
CONSTANTS = $(INPUT_DIR)/hardware/constant/7nm_default.yaml
SYS_CONFIG = $(INPUT_DIR)/software/system/sys_default.yaml

VERBOSE ?= false

OUTPUT_DIR = outputs
# Create output directory if it does not exist
$(OUTPUT_DIR):
	@mkdir $@ || :

# Hardware Exploration
define HW_GEN
$(hardware).hw: $(OUTPUT_DIR)/$(hardware)/$(hardware).pkl
$(OUTPUT_DIR)/$(hardware)/$(hardware).pkl: $(HARDWARE_INPUT_DIR)/$(hardware).yaml | $(OUTPUT_DIR) pyenv_exists
	@echo "Running hardware exploration for $(hardware)"
	@if [ "$(VERBOSE)" = "true" ]; then \
		$(VENV_PYTHON3) phases/hardware_exploration.py --config-file $$< --constants-file $(CONSTANTS) --results-dir $(OUTPUT_DIR) --verbose; \
	else \
		$(VENV_PYTHON3) phases/hardware_exploration.py --config-file $$< --constants-file $(CONSTANTS) --results-dir $(OUTPUT_DIR); \
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
$(OUTPUT_DIR)/$(hardware)/$(model).pkl: $(MODELS_INPUT_DIR)/$(model).yaml $(OUTPUT_DIR)/$(hardware)/$(hardware).pkl
	@if [ "$(VERBOSE)" = "true" ]; then \
		$(VENV_PYTHON3) phases/software_evaluation.py --model $$< --hardware $(OUTPUT_DIR)/$(hardware)/$(hardware).pkl --sys-config $(INPUT_DIR)/software/system/$(hardware).yaml --results-dir $(OUTPUT_DIR) --verbose; \
	else \
		$(VENV_PYTHON3) phases/software_evaluation.py --model $$< --hardware $(OUTPUT_DIR)/$(hardware)/$(hardware).pkl --sys-config $(INPUT_DIR)/software/system/$(hardware).yaml --results-dir $(OUTPUT_DIR); \
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
	@make hw_example.clean
	$(VENV_PYTHON3) main.py -hw $(HARDWARE_INPUT_DIR)/hw_example.yaml -m $(MODELS_INPUT_DIR)/gpt3.yaml -c $(CONSTANTS) -s $(SYS_CONFIG) -o $(OUTPUT_DIR)
	$(VENV_PYTHON3) tests/compare.py --hardware hw_example --model gpt3

# Clean all
clean:
	rm -rf $(OUTPUT_DIR)
