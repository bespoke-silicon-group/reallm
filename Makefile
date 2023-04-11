export PLOT_SCRIPTS_PATH = $(abspath ./plot)
export ASIC_CLOUD_SCRIPTS_PATH = $(abspath ./hardware_exploration/scripts)
export PYTHONPATH := ${PYTHONPATH}:$(PLOT_SCRIPTS_PATH):$(ASIC_CLOUD_SCRIPTS_PATH)

OBJ_DIR = results

# Make build directory, or do nothing if it already exists
$(OBJ_DIR):
	mkdir $@ || :

# Hardware Exploration
hardware-exploration: $(OBJ_DIR)/exploration.csv
hbm-exploration: $(OBJ_DIR)/HBM_chiplet.csv

$(OBJ_DIR)/exploration.csv: | $(OBJ_DIR) 
	python hardware_exploration/chiplet_system_gen.py --config 'exploration' --results-dir $(abspath $(OBJ_DIR))

$(OBJ_DIR)/HBM_chiplet.csv: | $(OBJ_DIR) 
	python hardware_exploration/chiplet_system_gen.py --config 'HBM_chiplet' --results-dir $(abspath $(OBJ_DIR))

# Software Evaluation
software-evaluation: software-evaluation-gpt2 software-evaluation-gpt3 software-evaluation-tnlg software-evaluation-palm
software-evaluation-gpt2: $(OBJ_DIR)/gpt2_all.csv 
software-evaluation-gpt3: $(OBJ_DIR)/gpt3_all.csv 
software-evaluation-tnlg: $(OBJ_DIR)/tnlg_all.csv 
software-evaluation-palm: $(OBJ_DIR)/palm_all.csv

hbm-software-evaluation: hbm-software-evaluation-gpt2 hbm-software-evaluation-gpt3 hbm-software-evaluation-tnlg hbm-software-evaluation-palm
hbm-software-evaluation-gpt2: $(OBJ_DIR)/gpt2_all.csv 
hbm-software-evaluation-gpt3: $(OBJ_DIR)/gpt3_all.csv 
hbm-software-evaluation-tnlg: $(OBJ_DIR)/tnlg_all.csv 
hbm-software-evaluation-palm: $(OBJ_DIR)/palm_all.csv

$(OBJ_DIR)/%_all.csv: $(OBJ_DIR)/exploration.csv 
	python software_evaluation/gen_mapping.py --model $* --hw-csv $< --results-dir $(abspath $(OBJ_DIR))

$(OBJ_DIR)/%_HBM_chiplet.csv: $(OBJ_DIR)/HBM_chiplet.csv 
	python software_evaluation/gen_mapping.py --model $* --hw-csv $< --results-dir $(abspath $(OBJ_DIR))

# Plot
$(OBJ_DIR)/models_df.pkl: $(OBJ_DIR)/exploration.csv
	python plot/plot.py --target 'models_df' --results-dir '$(OBJ_DIR)'

exploration-info: $(OBJ_DIR)/models_df.pkl
	python plot/plot.py --target 'exploration_info' > $(OBJ_DIR)/exploration_info.txt

opt-designs: $(OBJ_DIR)/models_df.pkl
	python plot/plot.py --target 'optimal_designs' > $(OBJ_DIR)/optimal_designs.txt

plot-all: $(OBJ_DIR)/models_df.pkl $(OBJ_DIR)/HBM_chiplet.csv 
	python plot/plot.py --target 'compare_gpu_tpu'
	python plot/plot.py --target 'asic_profit'
	python plot/plot.py --target 'compare_memory'
	python plot/plot.py --target 'design_space_exploration'
	python plot/plot.py --target 'chip_size'
	python plot/plot.py --target 'batch_size'
	python plot/plot.py --target 'design_choice'
	python plot/plot.py --target 'server_flexibility'

# Plot Targets
plot-gpu-tpu-compare: $(OBJ_DIR)/models_df.pkl
	python plot/plot.py --target 'compare_gpu_tpu'

plot-asic-profit: $(OBJ_DIR)/models_df.pkl
	python plot/plot.py --target 'asic_profit'

plot-design-space: $(OBJ_DIR)/models_df.pkl
	python plot/plot.py --target 'design_space_exploration'

plot-chip-size: $(OBJ_DIR)/models_df.pkl
	python plot/plot.py --target 'chip_size'

plot-batch-size: $(OBJ_DIR)/models_df.pkl
	python plot/plot.py --target 'batch_size'

plot-design-choice: $(OBJ_DIR)/models_df.pkl hbm-software-evaluation
	python plot/plot.py --target 'design_choice'

plot-server-flexibility: $(OBJ_DIR)/models_df.pkl
	python plot/plot.py --target 'server_flexibility'

plot-compare-memory: $(OBJ_DIR)/models_df.pkl
	python plot/plot.py --target 'compare_memory'

clean-pdf:
	-rm -rf $(OBJ_DIR)/*.pdf

clean:
	-rm -rf $(OBJ_DIR)

