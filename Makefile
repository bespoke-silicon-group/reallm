export MAGIC_NUMBERS_PATH = $(abspath ./chiplet_cloud_simulator_vlsi_numbers)
export MICRO_ARCH_PATH = $(abspath ./micro_arch_sim)
export STRUCTS_PATH = $(abspath ./structs)
export PLOT_SCRIPTS_PATH = $(abspath ./plot)
export PYTHONPATH := ${PYTHONPATH}:$(PLOT_SCRIPTS_PATH):$(STRUCTS_PATH):$(MAGIC_NUMBERS_PATH):$(MICRO_ARCH_PATH)

OBJ_DIR = results

# Make build directory, or do nothing if it already exists
$(OBJ_DIR):
	mkdir $@ || :

# Hardware Exploration
hardware-exploration: $(OBJ_DIR)/exploration.pkl $(OBJ_DIR)/exploration.csv
test-hardware-exploration: $(OBJ_DIR)/test.pkl $(OBJ_DIR)/test.csv

$(OBJ_DIR)/exploration.pkl: | $(OBJ_DIR)
	python hardware_exploration.py --config 'exploration' --results-dir $(abspath $(OBJ_DIR))
$(OBJ_DIR)/test.pkl: | $(OBJ_DIR)
	python hardware_exploration.py --config 'test' --results-dir $(abspath $(OBJ_DIR))

# Software Evaluation
software-evaluation: software-evaluation-gpt2 software-evaluation-megatron software-evaluation-gpt3 software-evaluation-gopher software-evaluation-mtnlg software-evaluation-bloom software-evaluation-palm software-evaluation-llama2
software-evaluation-gpt2: $(OBJ_DIR)/gpt2.pkl
software-evaluation-megatron: $(OBJ_DIR)/megatron.pkl
software-evaluation-gpt3: $(OBJ_DIR)/gpt3.pkl
software-evaluation-gopher: $(OBJ_DIR)/gopher.pkl
software-evaluation-mtnlg: $(OBJ_DIR)/mtnlg.pkl
software-evaluation-bloom: $(OBJ_DIR)/bloom.pkl
software-evaluation-palm: $(OBJ_DIR)/palm.pkl
software-evaluation-llama2: $(OBJ_DIR)/llama2.pkl


$(OBJ_DIR)/%.pkl: $(OBJ_DIR)/exploration.pkl
	python software_evaluation.py --model $* --hw-pkl $< --results-dir $(abspath $(OBJ_DIR))

# Plot
$(OBJ_DIR)/models_df.pkl: $(OBJ_DIR)/exploration.pkl
	python plot/plot.py --target 'models_df' --results-dir '$(OBJ_DIR)'
