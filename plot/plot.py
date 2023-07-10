import argparse
import pandas as pd
import pickle
from plot_utils import *

models_label = {'gpt2':  'GPT2-1.4B',
                'tnlg':  'Turing NLG-17B',
                'gpt3':  'GPT3-175B',
                'palm':  'PaLM-540B',
                }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=str, default='read_model')
    parser.add_argument('--results-dir', type=str, default='results')
    args = parser.parse_args()
    target = args.target
    results_dir = args.results_dir

    hw_csv = results_dir+"/exploration.csv"
    pkl_file = results_dir+'/models_df.pkl'

    if target == 'models_df':
        hw = pd.read_csv(hw_csv)
        dfs = {}
        for model in models_label:
            # all_csv = "software_evaluation/"+model+"_all.csv"
            all_csv = f"{results_dir}/{model}_all.csv"
            pd.set_option("display.max.columns", None)
            df = pd.read_csv(all_csv)
            df = pd.concat([hw.loc[map(lambda x: x-1, df["srv_id"].to_numpy().tolist())].reset_index(drop=True), df], axis=1)
            dfs[model] = df
        with open(pkl_file, 'wb') as f:
            pickle.dump(dfs, f)
    else:
        with open(pkl_file, 'rb') as f:
            dfs = pickle.load(f)

        if target == 'exploration_info':
            print_exploration_info(dfs, models_label)
        elif target == 'optimal_designs':
            print_optimal_designs(dfs, models_label, ctx_len=2048)
        elif target == 'compare_gpu_tpu':
            compare_gpu_tpu(dfs, plot_dir=results_dir)
        elif target == 'asic_profit':
            asic_profit(plot_dir=results_dir)
        elif target == 'design_space_exploration':
            design_space_exploration(dfs, models_label, plot_dir=results_dir)
        elif target == 'chip_size':
            chip_size(dfs, plot_dir=results_dir)
        elif target == 'batch_size':
            batch_size(dfs, models_label, plot_dir=results_dir)
        elif target == 'pipeline_size':
            p_sweep(dfs, models_label, plot_dir=results_dir)
        elif target == 'design_choice':
            hw = pd.read_csv(hw_csv)
            HBM_dfs = {}
            for model in models_label:
                # all_csv = "software_evaluation/"+model+"_HBM_chiplet.csv"
                all_csv = f"{results_dir}/{model}_HBM_chiplet.csv"
                pd.set_option("display.max.columns", None)
                df = pd.read_csv(all_csv)
                df = pd.concat([hw.loc[map(lambda x: x-1, df["srv_id"].to_numpy().tolist())].reset_index(drop=True), df], axis=1)
                HBM_dfs[model] = df
            design_choice(dfs, HBM_dfs, models_label, plot_dir=results_dir)
        elif target == 'server_flexibility':
            server_flexibility(dfs, models_label, plot_dir=results_dir)
        elif target == 'compare_memory':
            compare_memory(plot_dir=results_dir)
        else:
            print('Wrong Plot Task:', target)
    
