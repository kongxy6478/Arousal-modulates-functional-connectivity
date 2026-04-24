#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 16:55:56 2026

@author: kongxiangyu
"""
import os   
import pickle    
import sys
sys.path.extend([
    "/data/disk0/kongxiangyu/result_explanation/GS_new/FCw_coupling_project",
    '/data/disk0/kongxiangyu/result_explanation/GS_new/',
    '/data/disk0/kongxiangyu/result_explanation/code/util/',
    '/data/disk0/kongxiangyu/result_explanation/trans/',
    '/data/disk0/kongxiangyu/result_explanation/arousal',
])
import numpy as np
import pandas as pd
import seaborn as sns
from src.utils import load_as_pickle,save_as_pickle
from io_utils.load_data import load_labels, load_awake
from src.ET_trans import get_info_pupilsize_wrapper
from tqdm import tqdm
from scipy.stats import spearmanr
from src_local.method import compute_tvfc_groups
from analysis.awake_coupling import compute_tvFC_coupling
from analysis.edge_analysis_old import (
                                    run_kmeans_across_k,
                                    align_and_calculate_stability,
                                    # get_subject_split_data,
                                    align_labels_to_reference,
                                    run_split_half_analysis,
                                    run_comprehensive_similarity_analysis,
                                    filter_results_for_plotting,
                                    analyze_similarity_by_parameter,
                                    summary_noisy_data,
                                    )
# from analysis.edge_permutation import (#node_permutation_null,
#                                        #permute_edge_labels,
#                                        #generate_null_labels_nodeperm_per_subject,
#                                        #build_symmetric_null)
from my_code.analysis_cluster import plot_L_method, find_l_method_cutoff,kmeans_Visualization
from my_code.data_preprocessing import (optimized_assemble_to_flat,
                                        flat_to_assemble_matrix,
                                        unflatten_modes,
                                        matrix_to_flat_vector) 
from neuromaps.datasets import fetch_fslr
from matplotlib.colors import ListedColormap, LinearSegmentedColormap 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from visualization.plot_matrix import plot_half_heatmap


from analysis.edge_wrapper import (run_community_edge_full_pipeline,
                                   run_community_node_full_pipeline,
                                   run_community_coupling_full_pipeline,
                                   run_community_edge_part_pipeline,
                                   run_community_node_part_pipeline,
                                   run_community_coupling_part_pipeline)

from my_code.validation import (aggregate_ftype_session_to_dict,
                                prepare_plotting_data, 
                                plot_regression_comparison_line,
                                plot_comparison_matrices,
                                plot_split_half_reliability,
                                plot_L_method, add_mean_in_result,
                                draw_raincloud_plot,
                                analyze_LI_all_levels_stability,
                                analyze_stability_core_v1,
                                process_iteration_half_split,
                                plot_LI_stability,
                                analyze_subject_stability,
                                align_to_reference,
                                run_split_half_analysis,
                                plot_subject_stability_metrics,
                                plot_stability_validation)

from my_code.edge_analysis import get_k_config

colors_k5 = ['#2364AA','#3DA5D9','#FEC601','#EA7317','#F94144']
cmap_k5 = ListedColormap(colors_k5, name='my_cmap')
grad_cmaps_k5 = {f"grad_{i}": LinearSegmentedColormap.from_list(
    f"grad_{i}", [(0.0, '#ffffff'), (1.0, color)]) 
    for i, color in enumerate(colors_k5)}

colors_k6 = ['#2364AA','#3DA5D9','#73BFB8','#FEC601','#EA7317','#D64550']
cmap_k6 = ListedColormap(colors_k6, name='my_cmap')
grad_cmaps_k6 = {f"grad_{i}": LinearSegmentedColormap.from_list(
    f"grad_{i}", [(0.0, '#ffffff'), (1.0, color)]) 
    for i, color in enumerate(colors_k6)}  
 
colors_k7 = ['#2364AA','#3DA5D9','#73BFB8','#B9C35D','#FEC601','#EA7317','#D64550']
cmap_k7 = ListedColormap(colors_k7, name='my_cmap')
grad_cmaps_k7 = {f"grad_{i}": LinearSegmentedColormap.from_list(
    f"grad_{i}", [(0.0, '#ffffff'), (1.0, color)]) 
    for i, color in enumerate(colors_k7)}

net_yeo7  = ['DMN', 'FPN', 'LIMB', 'VAN', 'DAN', 'SMN', 'VIS']
net_yeo7x2  = ['L_DMN', 'L_FPN', 'L_LIMB', 'L_VAN','L_DAN', 'L_SMN', 'L_VIS',
               'R_DMN', 'R_FPN', 'R_LIMB', 'R_VAN', 'R_DAN', 'R_SMN', 'R_VIS']


working_path = '/data/disk0/kongxiangyu/result_explanation/trans'
save_path = '/data/disk0/kongxiangyu/result_explanation/arousal/data'
figure_path = '/data/disk0/kongxiangyu/result_explanation/arousal/results/review_figure_v0'
output_path = '/data/disk0/kongxiangyu/result_explanation/arousal/results/review_figure_k7'


REST_ET_info = get_info_pupilsize_wrapper('REST')
MOVIE_ET_info = get_info_pupilsize_wrapper('MOVIE')

REST_version  = 'v0330Yan2023_squ_rest'
MOVIE_version = 'v0522Yan2023_squ_movie'

sessions=list(REST_ET_info['IDRuns_group'].keys())
summary_noisy = summary_noisy_data(REST_ET_info, working_path, REST_version)
save_as_pickle(summary_noisy,'summary_noisy.pkl')

surfaces = fetch_fslr()
lh, rh = surfaces['inflated']
labels_dict = load_labels()


size_list = [35,60,90]
step_list = [1,5,10]
lag_list = [-4,-3,-2,-1,0,1,2,3,4]
for size in tqdm(size_list):
    for step in tqdm(step_list):
        compute_tvfc_groups(sessions,
                            IDRuns=REST_ET_info['IDRuns_group'],
                            win_size=size,
                            win_step=step,
                            save_path=save_path,
                            version=REST_version,
                            data_path=working_path,
                            verbose=True
                            )

compute_tvfc_groups(list(MOVIE_ET_info['IDRuns_group'].keys()),
                    IDRuns=MOVIE_ET_info['IDRuns_group'],
                    win_size=30,
                    win_step=5,
                    save_path=save_path,
                    version=MOVIE_version,
                    data_path=working_path,
                    verbose=True
                    )
        
awake_dict_lags = load_as_pickle('awake_dict_lags.pkl')
summary_noisy = load_as_pickle('summary_noisy.pkl')
awake_dict_lags = {}
temp_path = '/data/disk0/kongxiangyu/result_explanation/arousal/data/v0330Yan2023_squ_rest/'
for size in tqdm(size_list):
    for step in tqdm(step_list):  
        for t in tqdm(lag_list):
            current_key = f'size{size}_step{step}_lag{t}'
            if awake_dict_lags.get(current_key) is None:
                awake_dict_lags[current_key] = load_awake(
                    'REST',
                    size=size * 1000,
                    step=step * 1000,
                    lag=t
                )
            save_as_pickle(awake_dict_lags,'awake_dict_lags.pkl')
            tvFC_paths = {'LL':f'{temp_path}/tvfc_LL_node', 
                          'RR':f'{temp_path}/tvfc_RR_node',
                          'LR':f'{temp_path}/tvfc_LR_node', 
                          'RL':f'{temp_path}/tvfc_RL_node'}
                
            compute_tvFC_coupling(
                tvFC_paths,
                awake_dict_lags[f'size{size}_step{step}_lag{t}'],
                save_path,
                dataset=f'REST_size{size}_step{step}_lag{t}',
                prefix=f'size{size}_step{step}',
                n_jobs=60,
                summary_noisy=summary_noisy,
                use_confounds=True
                )  
current_key = f'size{size}_step{step}_lag0'
awake_MOVIE_lag0 = load_awake(
    'MOVIE',
    size=size * 1000,
    step=step * 1000,
    lag=0
)
temp_path = f'/data/disk0/kongxiangyu/result_explanation/arousal/data/{MOVIE_version}/'
tvFC_paths = {'LL':f'{temp_path}/tvfc_LL_node', 
              'RR':f'{temp_path}/tvfc_RR_node',
              'LR':f'{temp_path}/tvfc_LR_node', 
              'RL':f'{temp_path}/tvfc_RL_node'}
    
compute_tvFC_coupling(
    tvFC_paths,
    awake_MOVIE_lag0,
    save_path,
    dataset=f'MOVIE_size{size}_step{step}_lag0',
    prefix=f'size{size}_step{step}',
    n_jobs=60,
    use_confounds=False
    )              
            
   
K_range = range(2, 15)
save_dir = './kmeans_results'
os.makedirs(save_dir, exist_ok=True)

checkpoint_dir = 'kmeans_checkpoints_v0'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

summary_scores_dict = {}
summary_scores_dict_con = {}

for size in tqdm(size_list, desc='Processing Sizes'):
    for step in tqdm(step_list, desc='Processing Steps', leave=False):  
        for t in tqdm(lag_list, desc='Processing Lags', leave=False):            
            current_key = f'size{size}_step{step}_lag{t}'
            
            # 1. Define base tasks for all window sizes
            sub_tasks = [('_raw', '', summary_scores_dict, current_key)]
            
            # 2. Add confound-related tasks only when size is 30
            if size == 30:
                
                sub_tasks.append(('_raw', '', summary_scores_dict_con, current_key))

                sub_tasks.extend([
                    # Raw Confound tasks -> store in summary_scores_dict_con
                    ('_con1', '_confounds', summary_scores_dict_con, f'{current_key}_con1'),
                    ('_con2', '_confounds_noblinks', summary_scores_dict_con, f'{current_key}_con2'),
                    ('_con3', '_confounds_noblinksgs', summary_scores_dict_con, f'{current_key}_con3'),
                ])

            for file_suffix, p1, target_dict, dict_key in sub_tasks:
                # Construct unique checkpoint filename
                checkpoint_filename = f"{dict_key}{file_suffix}.pkl"
                checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)
                
                # Check if result already exists on disk
                if os.path.exists(checkpoint_path):
                    with open(checkpoint_path, 'rb') as f:
                        result = pickle.load(f)
                    target_dict[dict_key] = result
                else:
                    # Execute aggregation if checkpoint not found
                    arousal_tvFC_coupling = aggregate_ftype_session_to_dict(
                        sessions, data_path=save_path, 
                        prefix=f'REST_{current_key}_arousal_tvFC_coupling',
                        prefix1=p1
                    )
                    
                    # Run K-means clustering across specified range
                    result = run_kmeans_across_k(arousal_tvFC_coupling, K_range=K_range, n_jobs=15)
                    
                    # Store in target dictionary and save to disk
                    target_dict[dict_key] = result
                    with open(checkpoint_path, 'wb') as f:
                        pickle.dump(result, f)

with open('summary_scores_dict_complete.pkl', 'wb') as f:
    pickle.dump(summary_scores_dict, f)

with open('summary_scores_dict_con_complete.pkl', 'wb') as f:
    pickle.dump(summary_scores_dict_con, f)
            

summary_scores_dict = load_as_pickle(f'{output_path}/summary_scores_dict_complete.pkl')
summary_scores_dict_con = load_as_pickle(f'{output_path}/summary_scores_dict_con_complete.pkl')

# %% find best k
base_key = 'size30_step5_lag0'
base_coupling = load_as_pickle(f'./data/REST_{base_key}_arousal_tvFC_coupling.pkl')

base_coupling_matrix = np.moveaxis(unflatten_modes(base_coupling.T,400),2,0)
arousal_coupling_dict = {}
arousal_coupling_dict['cortex'] = base_coupling_matrix
base_result = run_kmeans_across_k(base_coupling, K_range=K_range, n_jobs=15)

kmeans_Visualization(base_result,K_range,output_path)               
                   
scores_name = 'inertias'                  
target_k = plot_L_method(base_result[scores_name],
                         K_range,
                         scores_name,
                         output_path)
print(target_k)

k_range = range(2, 15)
target_k = 7
k_index = list(K_range).index(target_k)      

# %% basic information
N=400
labels_edge = load_as_pickle(f'{output_path}/kmeans_result_k{target_k}_label_edge.pkl')
ci_matrix_null_dicts = load_as_pickle(f'{output_path}/ci_matrix_null_dicts_k{target_k}.pkl')
ci_matrix_null = ci_matrix_null_dicts['cortex'].transpose(1, 2, 0)
IDRuns = np.concat(list(REST_ET_info['IDRuns_group'].values()))
subject_labels = []
run_labels = []
for i,IDRun in enumerate(IDRuns):
    subject_labels.append(IDRun.split('_',1)[0])
    run_labels.append(IDRun.split('_',1)[1])
    
subject_info = pd.read_excel(f'{save_path}/20170313SortedTwins1.xlsx')
subject_unique = np.unique(subject_labels)

subject_info['Subject'] = subject_info['Subject'].astype(str)
subject_unique_str = [str(i) for i in subject_unique]

selected_info = subject_info[subject_info['Subject'].isin(subject_unique_str)][['Subject', 'Age_in_Yrs', 'Gender']]
age_mean = selected_info['Age_in_Yrs'].mean()
age_std = selected_info['Age_in_Yrs'].std()
selected_info['Gender']


run_counts = pd.Series(subject_labels).value_counts()
plt.figure(figsize=(10, 6))
sns.histplot(run_counts, discrete=True, color='skyblue', edgecolor='black')
plt.title('Distribution of Runs per Subject', fontsize=15)
plt.xlabel('Number of Runs', fontsize=12)
plt.ylabel('Number of Subjects', fontsize=12)
plt.xticks(range(int(run_counts.min()), int(run_counts.max()) + 1)) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


df_counts = run_counts.reset_index()
df_counts.columns = ['SubjectID', 'RunCount']
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=df_counts, x='RunCount', palette='viridis')
ax.set_ylim(0, 100)
plt.title('Count of Subjects by Number of Available Runs')
plt.xlabel('Number of Runs')
plt.ylabel('Count of Subjects')
plt.savefig(f'{output_path}/validation/run_countplot.png', dpi=300)
plt.show()


df_matrix = pd.DataFrame({'Subject': subject_labels, 'Run': run_labels})
df_matrix['Value'] = 1
pivot_table = df_matrix.pivot_table(index='Subject', columns='Run', values='Value', fill_value=0)
plt.figure(figsize=(8, 16)) 
sns.heatmap(pivot_table, cbar=False, cmap="YlGnBu", linewidths=0.5)
plt.title('Data Availability Matrix (Subject x Run)')
plt.savefig(f'{output_path}/validation/run_availability.png', bbox_inches='tight',dpi=600)
plt.show()

# %% split half analysis
# ---- community
stats, half_labels = run_split_half_analysis(base_coupling,  
                                             REST_ET_info['IDRuns_group'],
                                             labels_edge,
                                             target_k,
                                             iterations=500,
                                             n_jobs=50)


plot_split_half_reliability(stats, target_k, cmap_k7, output_path, '')

config_sh = {'mode': 'split-half', 'best_k': 7, 'prefix': 'part1'}
plot_stability_validation(stats, config_sh, cmap=cmap_k7, output_path=output_path)


sns.heatmap(flat_to_assemble_matrix(half_labels['labels1'][1]),vmin=1,cmap=cmap_k5)
# ---- LI
LI_obs_h1_edge_iters = {'inte':[],'segre':[]}
LI_sig_h1_edge_iters = {'inte':[],'segre':[]}
LI_obs_h2_edge_iters = {'inte':[],'segre':[]}
LI_sig_h2_edge_iters = {'inte':[],'segre':[]}

LI_h1_node_iters = {'inte':[],'segre':[]}
LI_h1_node_iters = {'inte':[],'segre':[]}
LI_h2_node_iters = {'inte':[],'segre':[]}
LI_h2_node_iters = {'inte':[],'segre':[]}

LI_h1_coupling_iters = {'inte':[],'segre':[]}
LI_h2_coupling_iters = {'inte':[],'segre':[]}

n_jobs = 50 
from joblib import Parallel, delayed
for h_idx, h_type in enumerate(['half1', 'half2']):
    print(f"Running Parallel Analysis for {h_type}...")
    
    curr_labels = half_labels[f'labels{h_idx+1}']
    curr_idx = stats[f'idx_half{h_idx+1}']
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_iteration_half_split)(
            i, h_type, curr_labels, curr_idx, base_coupling,
            ci_matrix_null_dicts, ci_matrix_null, labels_dict, output_path, N
        ) 
        for i in tqdm(range(500))
    )
    
    results.sort(key=lambda x: x['n_iter'])
    
    target_obs_edge = LI_obs_h1_edge_iters if h_type == 'half1' else LI_obs_h2_edge_iters
    target_sig_edge = LI_sig_h1_edge_iters if h_type == 'half1' else LI_sig_h2_edge_iters
    target_node = LI_h1_node_iters if h_type == 'half1' else LI_h2_node_iters
    target_coupling = LI_h1_coupling_iters if h_type == 'half1' else LI_h2_coupling_iters
    
    for r in results:
        target_obs_edge['inte'].append(r['obs_edge']['inte'])
        target_obs_edge['segre'].append(r['obs_edge']['segre'])
        target_sig_edge['inte'].append(r['sig_edge']['inte'])
        target_sig_edge['segre'].append(r['sig_edge']['segre'])
        
        target_node['inte'].append({'mean': r['node']['mean_inte'], 'padj_': r['node']['padj_inte']})
        target_node['segre'].append({'mean': r['node']['mean_segre'], 'padj': r['node']['padj_segre']})
        
        target_coupling['inte'].append({'slope': r['coupling']['inte']['slope_pair_comm'], 
                                       'strength': r['coupling']['inte']['strength_pair_comm']})
        target_coupling['segre'].append({'slope': r['coupling']['segre']['slope_pair_comm'], 
                                        'strength': r['coupling']['segre']['strength_pair_comm']})

save_as_pickle(LI_obs_h1_edge_iters,f'{output_path}/LI_obs_h1_edge_iters.pkl')
save_as_pickle(LI_sig_h1_edge_iters,f'{output_path}/LI_sig_h1_edge_iters.pkl')
save_as_pickle(LI_h1_node_iters,f'{output_path}/LI_h1_node_iters.pkl')
save_as_pickle(LI_h1_coupling_iters,f'{output_path}/LI_h1_coupling_iters.pkl')

save_as_pickle(LI_obs_h2_edge_iters,f'{output_path}/LI_obs_h2_edge_iters.pkl')
save_as_pickle(LI_sig_h2_edge_iters,f'{output_path}/LI_sig_h2_edge_iters.pkl')
save_as_pickle(LI_h2_node_iters,f'{output_path}/LI_h2_node_iters.pkl')
save_as_pickle(LI_h2_coupling_iters,f'{output_path}/LI_h2_coupling_iters.pkl')


LI_obs_edge = load_as_pickle(f'{output_path}/LI4_obs_k{target_k}.joblib')
LI_sig_edge = load_as_pickle(f'{output_path}/LI4_obs_sig_k{target_k}.joblib')
LI_obs_coupling = load_as_pickle(f'{output_path}/LI_coupling_k{target_k}.joblib')
LI_obs_node = load_as_pickle(f'{output_path}/REST_network_result_k{target_k}.joblib')

all_levels_obs_results = analyze_LI_all_levels_stability(
    LI_obs_edge, LI_obs_h1_edge_iters, LI_obs_h2_edge_iters,
    LI_obs_node, LI_h1_node_iters, LI_h2_node_iters,
    LI_obs_coupling, LI_h1_coupling_iters, LI_h2_coupling_iters,
    min_denominator=1e-4
)
all_levels_sig_results = analyze_LI_all_levels_stability(
    LI_sig_edge, LI_sig_h1_edge_iters, LI_sig_h2_edge_iters,
    LI_obs_node, LI_h1_node_iters, LI_h2_node_iters,
    LI_obs_coupling, LI_h1_coupling_iters, LI_h2_coupling_iters,
    min_denominator=1e-4
)

plot_LI_stability(all_levels_obs_results,
                  all_levels_sig_results,
                  net_yeo7,
                  cmap_k7,
                  output_path,
                  'split-half')
save_as_pickle(all_levels_obs_results,f'{output_path}/all_levels_obs_results.pkl')
save_as_pickle(all_levels_sig_results,f'{output_path}/all_levels_sig_results.pkl')

all_levels_obs_results = load_as_pickle(f'{output_path}/all_levels_obs_results.pkl')
all_levels_sig_results = load_as_pickle(f'{output_path}/all_levels_sig_results.pkl')

# %% across param similarity (size+step+lag)

target_k=7
data_dicts = [
    (summary_scores_dict_con,    "sim-con"),
    (summary_scores_dict,        "sim")
]
target_metric = 'dice'
com_keys = ['community_1','community_2','community_3','community_4',
            'community_5','community_6','community_7','mean']

for data_dict, name in data_dicts:
    
    file_path = f'{output_path}/{name}_k{target_k}.pkl'
    if not os.path.exists(file_path):
        res, res_net = run_comprehensive_similarity_analysis(data_dict, 
                                                             target_k, 
                                                             labels_dict,
                                                             net_yeo7x2,
                                                             K_range, 60)
        save_as_pickle({'edge':res, 'net':res_net}, file_path)
    else:
        res_dict = load_as_pickle(file_path)

        res_dict['net'] = add_mean_in_result(res_dict['net'])
        res_dict['edge'] = add_mean_in_result(res_dict['edge'])
       
    
    for level in ['edge']:

        if 'con' in name:
            res_level = filter_results_for_plotting(res_dict[level],None, ['_con2']) 
            con_suffix = '_con2'
            
        else:
            res_level = res_dict[level]
            con_suffix = ''
            
        for com_key in com_keys:
            
            sim_df = res_level[target_metric][com_key].copy()
    
            # across all
            if 'con' in name:
                plt.figure(figsize=(8, 6))
                g = sns.heatmap(sim_df, square=True, vmin=0, vmax=1)
                g.figure.savefig(f'{output_path}/png/{name}_across-all_{target_metric}_{com_key}_{level}{con_suffix}.png', dpi=600)
                
                row_mid = len(sim_df) // 2
                col_mid = len(sim_df.columns) // 2
                sim_df = sim_df.iloc[row_mid:, :col_mid]
            else:
                plt.figure(figsize=(8, 6))
                g = sns.heatmap(sim_df, square=True, vmin=0.5, vmax=1)
                g.figure.savefig(f'{output_path}/png/{name}_across-all_{target_metric}_{com_key}_{level}{con_suffix}.png', dpi=600)

            # plt.close() 
            
            # across lag
            sim_df_lag = analyze_similarity_by_parameter(sim_df,'lag')
            plt.figure(figsize=(8,6))
            g = sns.heatmap(sim_df_lag, square=True, linewidth=.5)
            g.figure.savefig(f'{output_path}/png/{name}_across-lag_{target_metric}_{com_key}_{level}{con_suffix}.png', dpi=600)
            
            # across step
            sim_df_step = analyze_similarity_by_parameter(sim_df,'step')
            plt.figure(figsize=(8,6))
            g = sns.heatmap(sim_df_step, square=True, annot=True, fmt=".3f", linewidth=.5)
            g.figure.savefig(f'{output_path}/png/{name}_across-step_{target_metric}_{com_key}_{level}{con_suffix}.png', dpi=600)
            
            # across size
            sim_df_size = analyze_similarity_by_parameter(sim_df,'size')
            plt.figure(figsize=(8,6))
            g = sns.heatmap(sim_df_size, square=True, annot=True, fmt=".3f", linewidth=.5)
            g.figure.savefig(f'{output_path}/png/{name}_across-size_{target_metric}_{com_key}_{level}{con_suffix}.png', dpi=600)


labels_edge = summary_scores_dict_con['size30_step5_lag0']['labels'][target_k-2]+1
labels_edge_con = summary_scores_dict_con['size30_step5_lag2_con2']['labels'][target_k-2]+1

labels_edge_con_aligned = align_labels_to_reference(labels_edge,labels_edge_con)

ci_matrix = flat_to_assemble_matrix(labels_edge)
ci_matrix[ci_matrix==0] = np.nan

ci_matrix_con_aligned =  flat_to_assemble_matrix(labels_edge_con_aligned)
ci_matrix_con_aligned[ci_matrix_con_aligned==0] = np.nan

print(align_and_calculate_stability(labels_edge,labels_edge_con_aligned))

plot_comparison_matrices(ci_matrix, 
                         ci_matrix_con_aligned, 
                         title1="original", 
                         title2="confounds", 
                         cmap=cmap_k7)    
    


# %% subject-level
          
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from scipy.optimize import linear_sum_assignment
result_subject = analyze_subject_stability(arousal_tvFC_coupling, 
                                           np.array(subject_labels), 
                                           labels_edge, target_k, 500, 30)
plot_subject_stability_metrics(result_subject,cmap_k7,output_path)
config_sbj = {'mode': 'subject', 'best_k': 7, 'prefix': 'main'}
plot_stability_validation(result_subject, config_sbj, cmap=cmap_k7, output_path=output_path)

# averaging 
sub_ids = np.array(subject_labels)
unique_subs = np.unique(sub_ids)
n_edges = arousal_tvFC_coupling.shape[1]
full_sub_matrix = np.zeros((len(unique_subs), n_edges))
for i, sub in enumerate(unique_subs):
    full_sub_matrix[i, :] = np.mean(arousal_tvFC_coupling[sub_ids == sub], axis=0)

kmeans_strat1 = KMeans(n_clusters=7, n_init=20, random_state=42).fit(full_sub_matrix.T)
strat1_aligned = align_to_reference(kmeans_strat1.labels_+1, labels_edge,7)

# result_subject['strat2_metrics']['aligned_labels']
sns.heatmap(flat_to_assemble_matrix(kmeans_strat1.labels_+1),vmin=1,cmap=cmap_k7)


LI_obs_sbj_edge_iters = {'inte':[],'segre':[]}
LI_sig_sbj_edge_iters = {'inte':[],'segre':[]}

LI_sbj_node_iters = {'inte':[],'segre':[]}
LI_sbj_node_iters = {'inte':[],'segre':[]}

LI_sbj_coupling_iters = {'inte':[],'segre':[]}
LI_sbj_coupling_iters = {'inte':[],'segre':[]}

n_jobs = 50 
sbj_labels = result_subject["strat2_metrics"]['all_edge_labels']
sbj_idx = result_subject["strat2_metrics"]['all_sampled_indices']
results_sbj = Parallel(n_jobs=n_jobs)(
    delayed(process_iteration_half_split)(
        i, 'sbj', sbj_labels, sbj_idx, base_coupling,
        ci_matrix_null_dicts, ci_matrix_null, labels_dict, output_path, N
    ) 
    for i in tqdm(range(500))
)
save_as_pickle(results_sbj,f'{output_path}/results_sbj.pkl')
results_sbj.sort(key=lambda x: x['n_iter'])

for r in results_sbj:
    LI_obs_sbj_edge_iters['inte'].append(r['obs_edge']['inte'])
    LI_obs_sbj_edge_iters['segre'].append(r['obs_edge']['segre'])
    LI_sig_sbj_edge_iters['inte'].append(r['sig_edge']['inte'])
    LI_sig_sbj_edge_iters['segre'].append(r['sig_edge']['segre'])
    
    LI_sbj_node_iters['inte'].append({'mean': r['node']['mean_inte'], 'padj_': r['node']['padj_inte']})
    LI_sbj_node_iters['segre'].append({'mean': r['node']['mean_segre'], 'padj': r['node']['padj_segre']})
    
    LI_sbj_coupling_iters['inte'].append({'slope': r['coupling']['inte']['slope_pair_comm'], 
                                          'strength': r['coupling']['inte']['strength_pair_comm']})
    LI_sbj_coupling_iters['segre'].append({'slope': r['coupling']['segre']['slope_pair_comm'], 
                                           'strength': r['coupling']['segre']['strength_pair_comm']})

save_as_pickle(LI_obs_sbj_edge_iters,f'{output_path}/LI_obs_sbj_edge_iters.pkl')
save_as_pickle(LI_sig_sbj_edge_iters,f'{output_path}/LI_sig_sbj_edge_iters.pkl')
save_as_pickle(LI_sbj_node_iters,f'{output_path}/LI_sbj_node_iters.pkl')
save_as_pickle(LI_sbj_coupling_iters,f'{output_path}/LI_sbj_coupling_iters.pkl')


all_levels_obs_sbj_results = analyze_LI_all_levels_stability(
    LI_obs_edge, LI_obs_sbj_edge_iters, LI_obs_sbj_edge_iters,
    LI_obs_node, LI_sbj_node_iters, LI_sbj_node_iters,
    LI_obs_coupling, LI_sbj_coupling_iters, LI_sbj_coupling_iters,
    min_denominator=1e-4
)

all_levels_sig_sbj_results = analyze_LI_all_levels_stability(
    LI_sig_edge, LI_sig_sbj_edge_iters, LI_sig_sbj_edge_iters,
    LI_obs_node, LI_sbj_node_iters, LI_sbj_node_iters,
    LI_obs_coupling, LI_sbj_coupling_iters, LI_sbj_coupling_iters,
    min_denominator=1e-4
)


plot_LI_stability(all_levels_obs_sbj_results,
                  all_levels_sig_sbj_results,
                  net_yeo7,
                  cmap_k7,
                  output_path,
                  'subject')
save_as_pickle(all_levels_obs_sbj_results,f'{output_path}/all_levels_obs_sbj_results.pkl')
save_as_pickle(all_levels_sig_sbj_results,f'{output_path}/all_levels_sig_sbj_results.pkl')

sns.heatmap(flat_to_assemble_matrix(result_subject['strat1_results']['aligned_labels']),cmap=cmap_k5)
# %% run main analysis   

n_net=7
n_comm=5
ci_network_MOVIE_dict = {'mean': {ftype: np.zeros((n_comm, n_net, n_net)) for ftype in ['LL', 'RR', 'LR', 'RL']}}
ci_network_MOVIE_dict['mean']['cortex'] = np.zeros((n_comm, n_net * 2, n_net * 2))
ci_network_dict = load_as_pickle(f'{output_path}/REST_ci_network_dict_k{n_comm}.joblib')   
r_movie = np.zeros(n_comm)
from analysis.network_mapping import aggregate_edge_to_network

# ---- 1. Configuration and Paths ---
# Use predefined colormaps and constants

N = 400
target_k = 7
n_perm = 10000
base_key = 'size30_step5_lag0'
colors_k, cmap_k, grad_cmaps_k = get_k_config(target_k)
# Define output and resource paths
output_path = f'/data/disk0/kongxiangyu/result_explanation/arousal/results/review_figure_k{target_k}'
nodeperm_dir = f"./nodeperm_labels_review_k{target_k}"
ci_dict_path = f'{output_path}/ci_matrix_dicts_k{target_k}.pkl'
ci_null_dict_path = f'{output_path}/ci_matrix_null_dicts_k{target_k}.pkl'

# Ensure output directory exists
os.makedirs(os.path.join(output_path, 'png'), exist_ok=True)

# ---- 2. Load or Compute Arousal Coupling Data ---
coupling_cache_file = f'{save_path}/aggregated_arousal_coupling.npy'

if os.path.exists(coupling_cache_file):
    print("Loading cached arousal coupling data...")
    arousal_coupling_matrix = np.load(coupling_cache_file)
else:
    print("Aggregating arousal coupling data...")
    arousal_tvFC_coupling = aggregate_ftype_session_to_dict(
        sessions, data_path=save_path, 
        prefix=f'REST_{base_key}_arousal_tvFC_coupling',
        prefix1=''
    )
    # Assemble flat vectors into matrices for all 485 subjects
    arousal_coupling_matrix = np.stack([
        flat_to_assemble_matrix(arousal_tvFC_coupling[i]) for i in range(485)
    ])
    np.save(coupling_cache_file, arousal_coupling_matrix)



# ---- 3. K-means Clustering and Community Matrix ---
# Check if K-means results are already computed
kmeans_result_path = f'{output_path}/kmeans_result_k{target_k}_label_edge.pkl'

if os.path.exists(kmeans_result_path):
    print("Loading K-means results...")
    labels_edge = load_as_pickle(kmeans_result_path)
else:
    print("Running K-means clustering...")
    # arousal_tvFC_coupling needs to be defined from previous step if not cached
    arousal_tvFC_coupling = aggregate_ftype_session_to_dict(
        sessions, data_path=save_path, 
        prefix=f'REST_{base_key}_arousal_tvFC_coupling',
        prefix1=''
    )
    result = run_kmeans_across_k(arousal_tvFC_coupling, K_range=range(2,15), n_jobs=15)
    labels_edge = result['labels'][target_k-2] + 1
    save_as_pickle(labels_edge,kmeans_result_path)


# Extract labels for the target K (index = target_k - 2)
ci_matrix = flat_to_assemble_matrix(labels_edge)
ci_matrix[ci_matrix == 0] = np.nan

# ---- 4. Visualization: Community Heatmap ---
plt.figure(figsize=(8, 6))
plt.imshow(ci_matrix, cmap=cmap_k,vmin=1)
cbar = plt.colorbar()
cbar.set_label('Communities ID', fontsize=16)
cbar.ax.tick_params(labelsize=14)
cbar.locator = ticker.MaxNLocator(integer=True)
cbar.update_ticks()

plt.title('Communities from arousal-FC coupling', fontsize=16)
plt.axhline(200-0.5, color='k', linewidth=2)
plt.axvline(200-0.5, color='k', linewidth=2)
plt.xticks([])  
plt.yticks([])
# plt.savefig(f'{output_path}/png/REST_coupling_community_k{target_k}.png', dpi=600)
plt.show()

# Plot individual community heatmaps
for target_k in [5,6,7,8,9]:
    labels_edge = result['labels'][target_k-2] + 1
    ci_matrix = flat_to_assemble_matrix(labels_edge)
    ci_matrix[ci_matrix == 0] = np.nan
    for i in range(target_k):
        ci_matrix_com = ci_matrix.copy()
        ci_matrix_com[ci_matrix_com != (i+1)] = np.nan
        ci_matrix_com[ci_matrix_com == (i+1)] = 1
        plt.figure(figsize=(6, 6))
        plt.imshow(ci_matrix_com, cmap=cmap_k,vmin=1)
        plt.axhline(200-0.5, color='k', linewidth=2)
        plt.axvline(200-0.5, color='k', linewidth=2)
        plt.xticks([])  
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f'{output_path}/validation/REST_ci-edge-cortex_community{i+1}_k{target_k}.png', dpi=600)
        plt.close()
        

# ---- 5. Generate and Save Community Dictionaries ---
if not os.path.exists(ci_dict_path):
    ci_matrix_dicts = {
        'cortex': ci_matrix,
        'LL': ci_matrix[:N//2, :N//2],
        'RR': ci_matrix[N//2:, N//2:],
        'LR': ci_matrix[:N//2, N//2:],
        'RL': ci_matrix[N//2:, :N//2]
    }
    save_as_pickle(ci_matrix_dicts, ci_dict_path)
else:
    ci_matrix_dicts = load_as_pickle(ci_dict_path)

# ---- 6. Null Models Processing (Node Permutation) ---
if os.path.exists(ci_null_dict_path):
    print("Loading cached null community dictionaries...")
    ci_matrix_null_dicts = load_as_pickle(ci_null_dict_path)
    # Reconstruct 3D null matrix for the node pipeline if needed
    ci_matrix_null = ci_matrix_null_dicts['cortex'].transpose(1, 2, 0)
else:
    print("Processing null models...")
    ci_matrix_null = np.zeros((N, N, n_perm))
    triu_idx = np.triu_indices(N, 1)
    
    for n in tqdm(range(n_perm), desc="Loading Null Labels"):
        null_label_file = f'{nodeperm_dir}/nodeperm_labels_{n:04d}.npy'
        if os.path.exists(null_label_file):
            labels_null = np.load(null_label_file)
            # Fill upper triangle and symmetrize
            temp_mat = np.zeros((N, N))
            temp_mat[triu_idx] = labels_null
            ci_matrix_null[:, :, n] = temp_mat + temp_mat.T
    
    ci_matrix_null[ci_matrix_null == 0] = np.nan

    # Organize into dictionaries for pipeline
    ci_matrix_null_dicts = {
        'cortex': np.moveaxis(ci_matrix_null, -1, 0), # [n_perm, N, N]
        'LL': np.moveaxis(ci_matrix_null[:N//2, :N//2, :], -1, 0),
        'RR': np.moveaxis(ci_matrix_null[N//2:, N//2:, :], -1, 0),
        'LR': np.moveaxis(ci_matrix_null[:N//2, N//2:, :], -1, 0),
        'RL': np.moveaxis(ci_matrix_null[N//2:, :N//2, :], -1, 0)
    }
    save_as_pickle(ci_matrix_null_dicts, ci_null_dict_path)

# ---- 7. Run Integrated Analysis Pipelines ---
print("Running full analysis pipelines...")

# Edge-level community analysis
run_community_edge_full_pipeline(
    ci_matrix_dicts,
    ci_matrix_null_dicts,
    labels_dict,  
    output_path, 
    'REST',
    grad_cmaps_k,
    cmap_k,
    colors_k,
    n_perm=n_perm, 
    plot_heatmap=True, 
    run_null=True
)

# Node-level brain mapping analysis
run_community_node_full_pipeline(
    ci_matrix,
    ci_matrix_null,
    labels_dict,
    lh, rh,
    output_path,
    'REST',
    grad_cmaps_k, 
    colors_k,
    cmap_k,
    n_perm=n_perm,
    plot_brain=True
)

# Coupling-specific analysis
run_community_coupling_full_pipeline(
    ci_matrix_dicts,
    arousal_coupling_matrix,
    labels_dict,
    grad_cmaps_k, 
    colors_k, 
    output_path, 
    'REST'
)
