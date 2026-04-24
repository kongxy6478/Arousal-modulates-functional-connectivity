#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:27:44 2026

@author: kongxiangyu
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from src.utils import load_as_pickle,save_as_pickle
from analysis.edge_wrapper import (run_community_edge_part_pipeline,
                                   run_community_node_part_pipeline,
                                   run_community_coupling_part_pipeline)
from my_code.data_preprocessing import (optimized_assemble_to_flat,
                                        flat_to_assemble_matrix,
                                        unflatten_modes) 
from visualization.plot_matrix import plot_half_heatmap

def aggregate_ftype_session_to_dict(sessions, 
                                    data_path, 
                                    prefix='REST_size60_step1_lag-4_arousal_tvFC_coupling',
                                    prefix1=None,
                                    N=400):
    from my_code.data_preprocessing import optimized_assemble_to_flat
                                        

    all_sessions_data = []
    import gc
    for ses in sessions:
    
        ll = np.stack(load_as_pickle(f'{data_path}/{prefix}_LL_{ses}{prefix1}.pkl'))
        rr = np.stack(load_as_pickle(f'{data_path}/{prefix}_RR_{ses}{prefix1}.pkl'))
        lr = np.stack(load_as_pickle(f'{data_path}/{prefix}_LR_{ses}{prefix1}.pkl'))
        rl = np.stack(load_as_pickle(f'{data_path}/{prefix}_RL_{ses}{prefix1}.pkl'))
        
        ll = ll.reshape(ll.shape[0],N//2,N//2)
        rr = rr.reshape(rr.shape[0],N//2,N//2)
        lr = lr.reshape(lr.shape[0],N//2,N//2)
        rl = rl.reshape(rl.shape[0],N//2,N//2)
        
        ses_flat = optimized_assemble_to_flat(ll, rr, lr, rl)
    
        del ll, rr, lr, rl
        gc.collect() 
        
        all_sessions_data.append(ses_flat)
        
    arousal_tvFC_edge_coupling = np.vstack(all_sessions_data)
    arousal_tvFC_coupling = all_sessions_data    
    save_as_pickle(arousal_tvFC_coupling,f'{data_path}/{prefix}{prefix1}.pkl')
    
    return arousal_tvFC_coupling


from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

def _compute_single_k(X_scaled, k, n_init=10, random_state=0):
    """
    Internal helper function to run KMeans and metrics for a single K.
    """
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    labels = km.fit_predict(X_scaled)
    
    # Calculate metrics
    inertia = km.inertia_
    
    # Note: silhouette_score is O(N^2), very slow for 79800 samples.
    # We use a sample (e.g., 10000) to speed up if necessary, 
    # but here we keep it full unless it's too slow.
    sil_avg = silhouette_score(X_scaled, labels, sample_size=10000) if X_scaled.shape[0] > 10000 else silhouette_score(X_scaled, labels)
    
    ch_score = calinski_harabasz_score(X_scaled, labels)
    db_index = davies_bouldin_score(X_scaled, labels)
    
    return {
        'k': k,
        'inertia': inertia,
        'silhouette': sil_avg,
        'ch': ch_score,
        'db': db_index,
        'labels': labels
    }

def run_kmeans_across_k(arousal_tvFC_coupling, K_range=range(2, 15), n_jobs=8):
    """
    Parallelized version of KMeans evaluation across K.
    
    Parameters
    ----------
    arousal_tvFC_coupling : array (n_runs, n_edges)
    K_range : iterable
    n_jobs : int, default=-1 (use all CPUs)
    """
    # 1. Preprocessing (Consistent with our previous discussion)
    X = arousal_tvFC_coupling.T  # Shape (79800, 485)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Starting parallel processing for K in {list(K_range)}...")
    
    # 2. Parallel Execution using joblib
    # n_jobs=-1 uses all available cores
    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_single_k)(X_scaled, k) for k in K_range#, desc="Dispatching jobs")
    )
    
    # 3. Reassemble results
    summary_scores = {
        'inertias': [],
        'silhouette_scores': [],
        'ch_scores': [],
        'db_scores': [],
        'labels': []
    }
    
    # Sort by K to ensure order
    results.sort(key=lambda x: x['k'])
    
    for res in results:
        summary_scores['inertias'].append(res['inertia'])
        summary_scores['silhouette_scores'].append(res['silhouette'])
        summary_scores['ch_scores'].append(res['ch'])
        summary_scores['db_scores'].append(res['db'])
        summary_scores['labels'].append(res['labels'])
        
    return summary_scores

    
from sklearn.metrics import jaccard_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix    
from scipy.optimize import linear_sum_assignment    
def align_and_calculate_stability(base_labels, target_labels, n_clusters):
    """
    Align target_labels to base_labels using the Hungarian Algorithm and 
    calculate Jaccard, Dice, ARI, and NMI stability for each community.

    Parameters
    ----------
    base_labels : array, shape (n_samples,)
        Reference labels (e.g., from original parameters).
    target_labels : array, shape (n_samples,)
        Labels to be aligned (e.g., from shifted parameters).
    n_clusters : int
        Number of clusters (K).

    Returns
    -------
    jaccard_indices : dict
    dice_indices : dict
    ari_indices : dict
    nmi_indices : dict
    aligned_labels : array
        The target_labels remapped to match the base_labels' indexing.
    """
    # 1. Compute Confusion Matrix (Intersections)
    contingency_matrix = np.zeros((n_clusters, n_clusters))
    for i in range(n_clusters):
        for j in range(n_clusters):
            # i: index for base, j: index for target
            intersect = np.sum((base_labels == i) & (target_labels == j))
            contingency_matrix[i, j] = intersect

    # 2. Hungarian Algorithm (Minimize cost = Maximize intersection)
    # We use -contingency_matrix because linear_sum_assignment finds minimum
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # 3. Create Mapping (target_label -> base_label)
    # col_ind[i] is the target label that best matches base label i
    mapping = {target_label: base_label for base_label, target_label in zip(row_ind, col_ind)}
    
    aligned_labels = np.array([mapping[label] for label in target_labels])

    # 4. Calculate Metrics for each Community
    jaccard_indices = {}
    dice_indices = {}
    ari_indices = {}
    nmi_indices = {}

    for k in range(n_clusters):
        # Mask for current community
        mask_base = (base_labels == k)
        mask_aligned = (aligned_labels == k)
        
        # --- Jaccard ---
        j_score = jaccard_score(mask_base, mask_aligned)
        jaccard_indices[f'community_{k+1}'] = j_score
        
        # --- Dice ---
        intersection = np.sum(mask_base & mask_aligned)
        size_base = np.sum(mask_base)
        size_aligned = np.sum(mask_aligned)
        d_score = (2.0 * intersection) / (size_base + size_aligned) if (size_base + size_aligned) > 0 else 0
        dice_indices[f'community_{k+1}'] = d_score
        
        # --- ARI ---
        a_score = adjusted_rand_score(mask_base, mask_aligned)
        ari_indices[f'community_{k+1}'] = a_score
        
        # --- NMI ---
        n_score = normalized_mutual_info_score(mask_base, mask_aligned)
        nmi_indices[f'community_{k+1}'] = n_score
        
    # Calculate means
    jaccard_indices['mean'] = np.mean(list(jaccard_indices.values()))
    dice_indices['mean'] = np.mean(list(dice_indices.values()))
    ari_indices['mean'] = np.mean(list(ari_indices.values()))
    nmi_indices['mean'] = np.mean(list(nmi_indices.values()))
    
    return jaccard_indices, dice_indices, ari_indices, nmi_indices

def get_subject_split_data(data_matrix, session_dict):
    """
    Groups indices by subject ID to ensure subject-level independent splitting.
    
    Returns:
    - subject_to_indices: Dictionary mapping sub_id to list of row indices
    - unique_subjects: List of unique subject IDs
    """
    import random
    
    subject_to_indices = {}
    current_idx = 0
    session_order = ['rfMRI_REST1_7T_PA', 'rfMRI_REST2_7T_AP', 'rfMRI_REST3_7T_PA', 'rfMRI_REST4_7T_AP']
    
    for session in session_order:
        if session in session_dict:
            for run_name in session_dict[session]:
                sub_id = run_name.split('_')[0]
                if sub_id not in subject_to_indices:
                    subject_to_indices[sub_id] = []
                subject_to_indices[sub_id].append(current_idx)
                current_idx += 1
    
    unique_subjects = list(subject_to_indices.keys())     
       
    random.shuffle(unique_subjects)
    half_point = len(unique_subjects) // 2
    
    # Map subject lists back to matrix row indices
    idx_half1 = [idx for sub in unique_subjects[:half_point] for idx in subject_to_indices[sub]]
    idx_half2 = [idx for sub in unique_subjects[half_point:] for idx in subject_to_indices[sub]]
    
    # Slicing the data matrix
    data_half1 = data_matrix[idx_half1, :]
    data_half2 = data_matrix[idx_half2, :]
    
    return data_half1, data_half2

def get_subject_split_indices(session_dict):
    """
    Step 1: Parse the session dictionary and group row indices by Subject ID.
    """
    subject_to_indices = {}
    current_idx = 0
    session_order = ['rfMRI_REST1_7T_PA', 'rfMRI_REST2_7T_AP', 'rfMRI_REST3_7T_PA', 'rfMRI_REST4_7T_AP']
    
    for session in session_order:
        if session in session_dict:
            for run_name in session_dict[session]:
                sub_id = run_name.split('_')[0]
                if sub_id not in subject_to_indices:
                    subject_to_indices[sub_id] = []
                subject_to_indices[sub_id].append(current_idx)
                current_idx += 1
                
    return subject_to_indices, list(subject_to_indices.keys())

def align_labels_to_reference(ref_labels, target_labels, k):
    """
    New Utility: Aligns target_labels to match ref_labels numerically.
    Uses Hungarian Algorithm to maximize the overlap between clusters.
    
    Parameters:
    - ref_labels: Reference cluster labels (1D array)
    - target_labels: Labels to be reordered (1D array)
    - k: Number of clusters
    
    Returns:
    - aligned_labels: target_labels mapped to ref_labels' indexing
    """
    # 1. Build contingency matrix (Jaccard similarity between all pairs)
    contingency = np.zeros((k, k))
    for i in range(k):
        mask_ref = (ref_labels == i)
        for j in range(k):
            mask_target = (target_labels == j)
            intersection = np.logical_and(mask_ref, mask_target).sum()
            union = np.logical_or(mask_ref, mask_target).sum()
            contingency[i, j] = intersection / union if union > 0 else 0

    # 2. Find optimal matching (maximize Jaccard overlap)
    # row_ind corresponds to ref_labels index, col_ind to target_labels index
    row_ind, col_ind = linear_sum_assignment(-contingency)
    
    # 3. Create mapping and reassign labels
    # mapping[original_target_label] = matched_ref_label
    mapping = {target_l: ref_l for ref_l, target_l in zip(row_ind, col_ind)}
    
    aligned_labels = np.array([mapping[l] for l in target_labels])
    
    return aligned_labels


def calculate_robust_stability(labels_ref, labels_target, k, return_matrix=False):
    """
    Step 3: Evaluate stability and alignment.
    """
    # 1. Hungarian Alignment for Jaccard
    jaccard_matrix = np.zeros((k, k))
    for i in range(k):
        mask_ref = (labels_ref == i)
        for j in range(k):
            mask_target = (labels_target == j)
            intersection = np.logical_and(mask_ref, mask_target).sum()
            union = np.logical_or(mask_ref, mask_target).sum()
            jaccard_matrix[i, j] = intersection / union if union > 0 else 0

    row_ind, col_ind = linear_sum_assignment(-jaccard_matrix)
    matched_jaccard = jaccard_matrix[row_ind, col_ind].mean()

    # 2. Global Metrics
    ari = adjusted_rand_score(labels_ref, labels_target)
    nmi = normalized_mutual_info_score(labels_ref, labels_target)
    
    if return_matrix:
        # Create a re-mapped version of target labels for confusion matrix visualization
        mapping = {old: new for new, old in zip(row_ind, col_ind)}
        aligned_target = np.array([mapping[l] for l in labels_target])
        conf_mat = confusion_matrix(labels_ref, aligned_target, normalize='true')
        return matched_jaccard, ari, nmi, conf_mat
        
    return matched_jaccard, ari, nmi

   
def split_data_by_subject(data_matrix, subject_to_indices, unique_subjects):
    """
    Step 2: Physically split the data matrix into two halves based on subject identity.
    """
    import random
    random.shuffle(unique_subjects)
    half_point = len(unique_subjects) // 2
    
    idx_half1 = [idx for sub in unique_subjects[:half_point] for idx in subject_to_indices[sub]]
    idx_half2 = [idx for sub in unique_subjects[half_point:] for idx in subject_to_indices[sub]]
    
    return data_matrix[idx_half1, :], data_matrix[idx_half2, :],idx_half1,idx_half2


def run_comprehensive_similarity_analysis(summary_scores_dict, target_k, k_range, n_jobs=20):
    """
    Optimized calculation of all-to-all similarity matrices using parallel processing.
    
    Args:
        summary_scores_dict: Dictionary containing 'labels' for different parameters.
        target_k: The specific number of clusters to analyze.
        k_range: The range of k values used in the original scoring.
        n_jobs: Number of CPU cores to use. -1 uses all available processors.
    """
    from itertools import combinations_with_replacement
    # Locating the index for the specific target_k in the k_range list
    k_index = list(k_range).index(target_k)
    keys = list(summary_scores_dict.keys())
    n_params = len(keys)
    
    # Pre-extract labels to avoid passing the massive dictionary to worker processes
    # This reduces IPC (Inter-Process Communication) overhead significantly
    all_labels = [summary_scores_dict[k]['labels'][k_index] for k in keys]
    
    # Define tracking keys for communities and the overall mean
    com_keys = [f'community_{i+1}' for i in range(target_k)] + ['mean']
    metrics_names = ['jaccard', 'dice', 'ari', 'nmi']
    
    # Generate unique pairs (i, j) for the upper triangle including the diagonal
    # This reduces calculations from N^2 to (N*(N+1))/2
    pairs = list(combinations_with_replacement(range(n_params), 2))
    
    print(f"Calculating parallel all-to-all similarity for K={target_k} using {n_jobs} cores...")

    # Helper function to process a single pair of label sets
    def compute_pair(i, j):
        # res: (jaccard_dict, dice_dict, ari_dict, nmi_dict, aligned_labels)
        res = align_and_calculate_stability(all_labels[i], all_labels[j], n_clusters=target_k)
        # Return indices and the first 4 dictionary results (metrics)
        return i, j, res[:4]

    # Execute parallel processing
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(compute_pair)(i, j) for i, j in tqdm(pairs, desc='Computing Pairs')
    )

    # Initialize empty matrices for each metric and community
    all_matrices = {
        m: {ck: np.zeros((n_params, n_params)) for ck in com_keys} 
        for m in metrics_names
    }

    # Populate matrices using the results from parallel workers
    for i, j, metrics_dicts in results_list:
        for m_idx, m_name in enumerate(metrics_names):
            current_metric_dict = metrics_dicts[m_idx]
            for ck in com_keys:
                val = current_metric_dict[ck]
                # Fill the upper triangle
                all_matrices[m_name][ck][i, j] = val
                # Mirror to the lower triangle if not on the diagonal
                if i != j:
                    all_matrices[m_name][ck][j, i] = val

    # Pack the result matrices into a nested dictionary of Pandas DataFrames
    results = {m: {} for m in metrics_names}
    for m in metrics_names:
        for ck in com_keys:
            results[m][ck] = pd.DataFrame(
                all_matrices[m][ck], 
                index=keys, 
                columns=keys
            )
    
    return results

def analyze_similarity_by_parameter(similarity_df, mode='lag'):

    def extract_param(name):
        parts = name.split('_')
        for p in parts:
            if mode in p:
                return p
        return 'unknown'

    temp_df = similarity_df.copy()
    temp_df['group'] = [extract_param(idx) for idx in temp_df.index]
    grouped_rows = temp_df.groupby('group').mean()
    
    grouped_rows = grouped_rows.T
    grouped_rows['group'] = [extract_param(idx) for idx in grouped_rows.index]
    param_sim_matrix = grouped_rows.groupby('group').mean()
    
    def sort_key(s):
        import re

        match = re.search(r"(-?\d+)", s)
        return int(match.group(1)) if match else 0
            
    sorted_index = sorted(param_sim_matrix.index, key=sort_key)
    param_sim_matrix = param_sim_matrix.reindex(index=sorted_index, columns=sorted_index)
    
    return param_sim_matrix

def summary_noisy_data(info, working_path, version):
    sessions=list(info['IDRuns_group'].keys())
    IDRuns = info['IDRuns_group']
    ET_group = info['ET_group_1hz']
    summary_noisy = {'motion':{session:np.zeros((24,900,IDRuns[session].shape[0])) 
                           for session in sessions},
                     'blinks':{session:np.zeros((900,IDRuns[session].shape[0])) 
                               for session in sessions},
                     'GS':{session:np.zeros((900,IDRuns[session].shape[0])) 
                           for session in sessions},
                     }
    image_path = '/data/disk0/kongxiangyu/image/32k/'
    for session in sessions:
        for i, IDRun in enumerate(IDRuns[session]):
            ID = IDRun[0:6]
            
            # movement
            movement = np.loadtxt(f'{image_path}/{ID}/{session}/Movement_Regressors.txt')
            movement_sq = movement**2
            movement_friston24 = np.hstack([movement, movement_sq])
            summary_noisy['motion'][session][:,:,i] = movement_friston24.T
            
        # blinks
        summary_noisy['blinks'][session] = np.isnan(ET_group[session])
        
        # GS
        cortex_node_ts_group = load_as_pickle(os.path.join(
            working_path, f'result/{version}/cortex_node_ts_group_{session}.joblib'))
        for i, IDRun in enumerate(IDRuns[session]):
            ID = IDRun[0:6]
            summary_noisy['GS'][session][:,i] = np.mean(cortex_node_ts_group[IDRun],axis=1)
            
    return summary_noisy

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
def plot_L_method(metrics, K_range, metrics_name='SSE', output_path='.'):
    """
    Automatically finds optimal K and plots:
    1. L-method regression fits.
    2. Total RMSE for each candidate K to prove why the best K was chosen.
    """
    k_values = np.array(list(K_range))
    metrics = np.array(metrics)
    k_values_reshaped = k_values.reshape(-1, 1)
    
    # --- Step 1: Search for Optimal K by minimizing Total RMSE ---
    # We need at least 2 points for each line (left and right)
    # search_space stores the indices of candidate 'elbow' points
    search_indices = range(1, len(k_values) - 1) 
    rmse_results = []
    candidate_ks = []

    for i in search_indices:
        # Define current candidate K
        curr_k = k_values[i]
        candidate_ks.append(curr_k)
        
        # Split data at index i (including curr_k in both for the elbow)
        x_left, y_left = k_values_reshaped[:i+1], metrics[:i+1]
        x_right, y_right = k_values_reshaped[i:], metrics[i:]
        
        # Fit linear models
        reg_left = LinearRegression().fit(x_left, y_left)
        reg_right = LinearRegression().fit(x_right, y_right)
        
        # Calculate RMSE for both sides
        mse_left = mean_squared_error(y_left, reg_left.predict(x_left))
        mse_right = mean_squared_error(y_right, reg_right.predict(x_right))
        
        # Calculate Total Weighted RMSE
        n_total = len(metrics)
        total_rmse = (len(y_left) * np.sqrt(mse_left) + len(y_right) * np.sqrt(mse_right)) / n_total
        rmse_results.append(total_rmse)
    
    # Identify the best K
    best_idx = np.argmin(rmse_results)
    optimal_k = candidate_ks[best_idx]
    
    # --- Step 2: Visualization ---
    png_path = os.path.join(output_path, 'png')
    os.makedirs(png_path, exist_ok=True)

    # FIGURE 1: Regression Fitting
    plt.figure(figsize=(10, 6))
    split_i = search_indices[best_idx]
    
    # Re-run best fit for plotting
    rl = LinearRegression().fit(k_values_reshaped[:split_i+1], metrics[:split_i+1])
    rr = LinearRegression().fit(k_values_reshaped[split_i:], metrics[split_i:])
    
    plt.plot(k_values, metrics, 'ko-', label=f'Original {metrics_name}', alpha=0.4)
    plt.plot(k_values[:split_i+1], rl.predict(k_values_reshaped[:split_i+1]), 'r--', linewidth=2, label='Left Fit')
    plt.plot(k_values[split_i:], rr.predict(k_values_reshaped[split_i:]), 'b--', linewidth=2, label='Right Fit')
    plt.axvline(x=optimal_k, color='green', linestyle=':', label=f'Optimal K={optimal_k}')
    
    plt.title(f'L-Method: Optimal K Search ({metrics_name})', fontsize=15)
    plt.xlabel('Number of Clusters (K)', fontsize=12)
    plt.ylabel('Metric Value', fontsize=12)
    plt.xticks(K_range)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(png_path, f'L_method_fit_{metrics_name}.png'), dpi=300)
    plt.show()

    plt.figure(figsize=(12, 6))
    k_labels = [str(x) for x in candidate_ks]
    bars = plt.bar(k_labels, rmse_results, color='lightgray', edgecolor='black', alpha=0.6)
    
    # Highlight the minimum RMSE bar
    bars[best_idx].set_color('mediumseagreen')
    bars[best_idx].set_edgecolor('darkgreen')
    bars[best_idx].set_alpha(0.9)
    
    # Add labels on top of bars
    for i, val in enumerate(rmse_results):
        plt.text(i, val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.title(f'L-Method Evaluation: Total RMSE per Candidate K', fontsize=15)
    plt.xlabel('Candidate Elbow Point (K)', fontsize=12)
    plt.ylabel('Total RMSE (Lower is Better)', fontsize=12)
    plt.axhline(min(rmse_results), color='red', linestyle='--', alpha=0.3)

    plt.savefig(os.path.join(png_path, f'L_method_rmse_eval_{metrics_name}.png'), dpi=300)
    plt.show()

    return optimal_k
def parse_param_string(param_name):
    """
    Parse parameter string.
    Naming convention: 'size30_step5_lag0' or 'size30_step5_lag0_con1'
    """
    if 'con' in param_name:
        parts = param_name.split('_')
        # Format: sizeXX_stepXX_lagXX_conX
        size_param, step_param, lag_param, confound = parts[0], parts[1], parts[2], parts[3]
        
        # Mapping according to definitions: con1=Full, con2=Medium, con3=Basic
        if confound.endswith('1'):
            con_param = 'Motion+Blinks+GS'
        elif confound.endswith('2'):
            con_param = 'Motion+GS'
        elif confound.endswith('3'):
            con_param = 'Motion'
        else:
            con_param = confound
    else: 
        parts = param_name.split('_')
        step_param, lag_param = parts[1], parts[2]
        con_param = 'Raw'

    return step_param, lag_param, con_param

def prepare_plotting_data(sim_results, target_metric='jaccard', target_com='mean'):
    """
    Convert results to long-form DataFrame, keeping only Confound vs. Raw similarity.
    """
    df_matrix = sim_results[target_metric][target_com]
    plot_rows = []
    
    all_keys = df_matrix.index.tolist()

    for param_name in all_keys:
        step, lag, con = parse_param_string(param_name)
        
        # Only calculate score for Confound versions relative to Raw, exclude Raw self-comparison
        if con != "Raw":
            base_key = f"size30_{step}_{lag}"
            if base_key in df_matrix.columns:
                score = df_matrix.loc[param_name, base_key]
                
                plot_rows.append({
                    "Step": step,
                    "Lag": lag,
                    "Confound_Version": con,
                    "Similarity": score
                })
        
    return pd.DataFrame(plot_rows)

def plot_regression_comparison_line(plot_df, output_path, target_step='step5',prefix=''):
    """
    Plot line chart: X-axis as Lag, colors representing different regression versions.
    Excludes Raw self-line to focus on robustness contrast.
    """
    import re
    # Filter by specific Step
    sub_df = plot_df[plot_df['Step'] == target_step].copy()
    
    # Ensure Lag is sorted numerically (handles strings like 'lag0', 'lag1')
    if sub_df['Lag'].dtype == object:
        sub_df['lag_val'] = sub_df['Lag'].apply(lambda x: int(re.search(r'(-?\d+)', x).group(1)))
        sub_df = sub_df.sort_values('lag_val')

    plt.figure(figsize=(10, 7))
    sns.set_theme(style="ticks", context="talk")

    # Plotting
    ax = sns.lineplot(
        data=sub_df, 
        x="Lag", 
        y="Similarity", 
        hue="Confound_Version", 
        style="Confound_Version",
        markers=True, 
        dashes=False, 
        markersize=10,
        linewidth=2.5,
        palette="flare"
    )
    
    plt.title(f"Stability: Confound Regressed vs. Raw ({target_step})", pad=60)
    plt.xlabel("Temporal Lag (TR)")
    plt.ylabel("Similarity to Raw Version")
    # plt.ylim(0.2, 1) 
    plt.grid(True, axis='y', alpha=0.3)
    
    n_versions = sub_df['Confound_Version'].nunique()
    plt.legend(
        loc='lower center', 
        bbox_to_anchor=(0.5, 1.01), 
        ncol=n_versions, 
        frameon=False,      
        fontsize='small', 
        handletextpad=0.5, 
        columnspacing=1.5
    )
    
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(f'{output_path}/png/Confound_Regressed_Stability{prefix}_{target_step}.png', dpi=600, bbox_inches='tight')
    plt.show()
    
def plot_comparison_matrices(matrix1, matrix2, title1="Original", title2="Aligned", cmap='viridis'):
    """
    Helper function to plot two matrices side-by-side for comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    im1 = axes[0].imshow(matrix1, cmap=cmap)
    axes[0].set_title(title1)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(matrix2, cmap=cmap)
    axes[1].set_title(title2)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im2, ax=axes[1])
    
    plt.tight_layout()
    plt.show()
    
def add_mean_in_result(res_level, selected_communities=['community_2', 'community_3', 'community_4']):

    updated_results = {}
    
    for metric_name, communities_dict in res_level.items():

        updated_results[metric_name] = communities_dict.copy()
        
        target_dfs = []
        for ck in selected_communities:
            if ck in communities_dict:
                target_dfs.append(communities_dict[ck])
            else:
                print(f"Warning: {ck} not found in metric {metric_name}")
        
        if target_dfs:

            combined_arrays = np.stack([df.values for df in target_dfs], axis=0)
            mean_array = np.mean(combined_arrays, axis=0)
            

            mean_df = pd.DataFrame(
                mean_array, 
                index=target_dfs[0].index, 
                columns=target_dfs[0].columns
            )
            
            updated_results[metric_name]['selected-mean'] = mean_df
            
    return updated_results    

def process_single_k(k, data1, data2, iter_idx):
    """
    Processes a single K value: clustering, explicit label alignment, 
    reliability metrics calculation, and returning labels.
    """
    samples1 = data1.T
    samples2 = data2.T
    
    # 1. Independent Clustering
    # Using iter_idx to ensure different seeds for each iteration
    km1 = KMeans(n_clusters=k, n_init=10, random_state=iter_idx * 100 + 42).fit(samples1)
    labels1 = km1.labels_  # Reference labels
    
    km2 = KMeans(n_clusters=k, n_init=10, random_state=iter_idx * 100 + 43).fit(samples2)
    labels2 = km2.labels_  # Labels to be aligned
    
    # 2. Label Alignment
    # Based on the overlap of samples between the two clusterings
    contingency_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            # Calculate the number of overlapping samples between cluster i (labels1) and cluster j (labels2)
            contingency_matrix[i, j] = np.sum((labels1 == i) & (labels2 == j))
    
    # Use Hungarian algorithm to find optimal matching (minimize negative overlap = maximize overlap)
    row_idx, col_idx = linear_sum_assignment(-contingency_matrix)
    
    # Map labels2 categories to labels1's coordinate system
    mapping = {old_label: new_label for new_label, old_label in zip(row_idx, col_idx)}
    mapped_labels2 = np.array([mapping[l] for l in labels2])
    
    # 3. Calculate Similarity Metrics
    ari = adjusted_rand_score(labels1, mapped_labels2)
    nmi = normalized_mutual_info_score(labels1, mapped_labels2)
    
    # Calculate Dice coefficient (overlap after alignment)
    def calculate_dice(l1, l2, num_k):
        dices = []
        for i in range(num_k):
            mask1 = (l1 == i)
            mask2 = (l2 == i)
            intersection = np.sum(mask1 & mask2)
            dice = (2. * intersection) / (np.sum(mask1) + np.sum(mask2) + 1e-10)
            dices.append(dice)
        return np.mean(dices), dices

    avg_dice, per_cluster_dice = calculate_dice(labels1, mapped_labels2, k)
    
    # 4. Normalized Confusion Matrix
    conf_mat = confusion_matrix(
        labels1, 
        mapped_labels2, 
        labels=range(k), 
        normalize='true'
    )
    
    return {
        'k': k,
        'ari': ari,
        'nmi': nmi,
        'dice': avg_dice,
        'per_cluster_dice': per_cluster_dice,
        'conf_mat': conf_mat,
        'labels_ref': labels1,
        'labels_aligned': mapped_labels2
    }

def process_iteration(iter_idx, data_matrix, subject_to_indices, unique_subjects, k_range):
    """
    Task for a single iteration: split data and compute all K values.
    This is the primary unit for parallelization.
    """
    # 1. Split data by subject (performed inside each process for unique splits)
    data1, data2 = split_data_by_subject(data_matrix, subject_to_indices, unique_subjects)
    
    # 2. Sequential loop over k_range within this iteration
    iter_results = []
    for k in k_range:
        res = process_single_k(k, data1, data2, iter_idx)
        iter_results.append(res)
    
    return iter_results

def run_split_half_analysis(data_matrix, session_dict, k_range=range(2, 15), iterations=20, n_jobs=15):
    """
    Main orchestration function: performs split-half reliability analysis in parallel.
    Parallelization is executed over the 'iterations' level.
    """
    # Get subject-level split indices
    subject_to_indices, unique_subjects = get_subject_split_indices(session_dict)
    
    print(f"Starting parallel analysis: Total iterations={iterations}, Jobs={n_jobs}...")

    # CORE PARALLEL STEP: Dispatch iterations to multiple workers
    all_iterations_results = Parallel(n_jobs=n_jobs)(
        delayed(process_iteration)(i, data_matrix, subject_to_indices, unique_subjects, k_range)
        for i in tqdm(range(iterations),desc='running iterations ...')
    )

    # Initialize result storage
    stats = {
        k: {m: [] for m in ['dice', 'ari', 'nmi', 'per_cluster_dice', 'idx_half1', 'idx_half2']} 
        for k in k_range
    }
    conf_mat = {}
    half_labels = {}

    # Aggregating results from parallel workers
    for i, iter_res in enumerate(all_iterations_results):
        # We designate the result from the "last" index in the returned list as the sample for visualization
        is_last_item = (i == len(all_iterations_results) - 1)
        
        for res in iter_res:
            k = res['k']
            # Basic statistics
            for metric in ['dice', 'ari', 'nmi', 'per_cluster_dice', 'idx_half1', 'idx_half2']:
                stats[k][metric].append(res[metric])
            
            # Capture the data from the last iteration for downstream visualization/debug
            if is_last_item:
                conf_mat[k] = res['conf_mat']
                half_labels[k] = {
                    'labels1': res['labels_ref'],
                    'labels2': res['labels_aligned']
                }

    print(f"Analysis finished. Captured labels for K range: {list(half_labels.keys())}")
    return stats, conf_mat, half_labels



def align_to_reference(labels, ref_labels, k):
    """
    Align current clustering labels to a fixed reference using the Hungarian algorithm.
    """
    offset = 1
    # Create a contingency matrix (Reference x Current)
    contingency_matrix = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            # i = reference cluster index, j = current cluster index
            contingency_matrix[i, j] = np.sum((ref_labels == (i+offset)) & (labels == (j+offset)))
    
    # Use Hungarian algorithm to maximize overlap (minimize negative overlap)
    row_idx, col_idx = linear_sum_assignment(-contingency_matrix)
    
    # Create mapping: current_label -> reference_label
    # col_idx contains the original labels, row_idx contains their new aligned identities
    mapping = {old_idx + offset: new_idx + offset for new_idx, old_idx in zip(row_idx, col_idx)}
    
    # Transform labels
    aligned_labels = np.array([mapping.get(l, l) for l in labels])
    return aligned_labels

def process_single_k(k, data1, data2, ref_labels, iter_idx):
    """
    Processes a single K: Align both halves to a common reference.
    """
    # 1. Independent Clustering
    km1 = KMeans(n_clusters=k, n_init=10, random_state=iter_idx * 100 + 42).fit(data1.T)
    km2 = KMeans(n_clusters=k, n_init=10, random_state=iter_idx * 100 + 43).fit(data2.T)
    
    # 2. Dual Alignment to Reference
    # Both sets of labels are now in the "Reference Label Space"
    mapped_labels1 = align_to_reference(km1.labels_, ref_labels, k)
    mapped_labels2 = align_to_reference(km2.labels_, ref_labels, k)
    
    # 3. Calculate Similarity Metrics between the two aligned halves
    ari = adjusted_rand_score(mapped_labels1, mapped_labels2)
    nmi = normalized_mutual_info_score(mapped_labels1, mapped_labels2)
    
    # Calculate Dice (Consistency between aligned halves)
    def calculate_dice(l1, l2, num_k):
        dices = []
        for i in range(num_k):
            mask1 = (l1 == i)
            mask2 = (l2 == i)
            intersection = np.sum(mask1 & mask2)
            denom = np.sum(mask1) + np.sum(mask2)
            dice = (2. * intersection) / (denom + 1e-10)
            dices.append(dice)
        return np.mean(dices), dices

    avg_dice, per_cluster_dice = calculate_dice(mapped_labels1, mapped_labels2, k)
    
    # 4. Confusion Matrix (How well Half 2 matches Half 1 after global alignment)
    conf_mat = confusion_matrix(
        mapped_labels1, 
        mapped_labels2, 
        labels=range(k), 
        normalize='true'
    )
    
    return {
        'k': k,
        'ari': ari,
        'nmi': nmi,
        'dice': avg_dice,
        'per_cluster_dice': per_cluster_dice,
        'conf_mat': conf_mat,
        'labels_h1': mapped_labels1,
        'labels_h2': mapped_labels2
    }

def process_iteration(iter_idx, data_matrix, subject_to_indices, unique_subjects, best_k, reference_labels):
    """
    Task for a single iteration: Split data and align both halves to a global reference.
    """
    # 1. Split data by subject (Half-split)
    # Ensure this function is defined in your environment
    data1, data2, idx_half1, idx_half2 = split_data_by_subject(data_matrix, 
                                                               subject_to_indices, 
                                                               unique_subjects)
    
    # 2. Independent Clustering for each half
    # Use distinct seeds for each half and each iteration to ensure variability
    km1 = KMeans(n_clusters=best_k, n_init=10, random_state=iter_idx * 100 + 1).fit(data1.T)
    km2 = KMeans(n_clusters=best_k, n_init=10, random_state=iter_idx * 100 + 2).fit(data2.T)
    
    label1 = km1.labels_+1
    label2 = km2.labels_+1
    
    # 3. Align BOTH halves to the GLOBAL REFERENCE
    # Now both label sets share the same "physical meaning" defined by reference_labels
    aligned_labels1 = align_to_reference(label1, reference_labels, best_k)
    aligned_labels2 = align_to_reference(label2, reference_labels, best_k)
    # aligned_labels1=align_labels_to_reference(label1, labels_edge)
    # 4. Calculate Reliability Metrics between the two aligned halves
    ari = adjusted_rand_score(aligned_labels1, aligned_labels2)
    nmi = normalized_mutual_info_score(aligned_labels1, aligned_labels2)
    
    # Function to calculate Dice coefficient for each cluster
    def calculate_dice(l1, l2, k):
        dices = []
        for i in range(k):
            mask1 = (l1 == (i+1))
            mask2 = (l2 == (i+1))
            intersection = np.sum(mask1 & mask2)
            sum_val = np.sum(mask1) + np.sum(mask2)
            dice = (2. * intersection) / (sum_val + 1e-10)
            dices.append(dice)
        return np.mean(dices), dices

    avg_dice, per_cluster_dice = calculate_dice(aligned_labels1, aligned_labels2, best_k)
    # ci_matrix = flat_to_assemble_matrix(aligned_labels1)
    # 5. Normalized Confusion Matrix (Half 1 as row, Half 2 as column)
    conf_mat_h1h2 = confusion_matrix(
        aligned_labels1, 
        aligned_labels2, 
        labels=range(1,best_k+1), 
        normalize='true'
    )
    conf_mat_origh1 = confusion_matrix(
        reference_labels, 
        aligned_labels1, 
        labels=range(1,best_k+1), 
        normalize='true'
    )
    conf_mat_origh2 = confusion_matrix(
        reference_labels, 
        aligned_labels2, 
        labels=range(1,best_k+1), 
        normalize='true'
    )
    
    return {
        'ari': ari,
        'nmi': nmi,
        'dice': avg_dice,
        'per_cluster_dice': per_cluster_dice,
        'conf_mat_h1h2': conf_mat_h1h2,
        'conf_mat_origh1': conf_mat_origh1,
        'conf_mat_origh2': conf_mat_origh2,
        'labels_h1': aligned_labels1,
        'labels_h2': aligned_labels2,
        'idx_half1': idx_half1,
        'idx_half2': idx_half2
    }

def run_split_half_analysis(data_matrix, session_dict, reference_labels, best_k, iterations=100, n_jobs=15):
    """
    Main orchestration function for fixed-K split-half reliability analysis.
    
    Args:
        data_matrix: Voxel/Vertex by Time/Subject matrix.
        session_dict: Metadata for splitting.
        reference_labels: The "Gold Standard" labels (e.g., from full-sample clustering).
        best_k: The fixed number of clusters to analyze.
        iterations: Number of random splits.
        n_jobs: Parallel workers.
    """
    # Get subject-level split indices
    subject_to_indices, unique_subjects = get_subject_split_indices(session_dict)
    
    print("Starting Fixed-K Reliability Analysis...")
    print(f"K = {best_k}, Iterations = {iterations}, Reference-based Alignment.")

    # Parallel execution over iterations
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_iteration)(
            i, data_matrix, subject_to_indices, unique_subjects, best_k, reference_labels
        )
        for i in tqdm(range(iterations), desc='Processing iterations')
    )

    # Initialize storage for aggregated statistics
    stats = {
            'dice': [],
            'ari': [],
            'nmi': [],
            'per_cluster_dice': [],
            'idx_half1':[],
            'idx_half2':[],
            'conf_mat_h1h2':[],
            'conf_mat_origh1':[],
            'conf_mat_origh2':[],
    }
    
    # Extract results
    all_conf_mats = []
    for res in results:
        stats['dice'].append(res['dice'])
        stats['ari'].append(res['ari'])
        stats['nmi'].append(res['nmi'])
        stats['per_cluster_dice'].append(res['per_cluster_dice'])
        stats['idx_half1'].append(res['idx_half1'])
        stats['idx_half2'].append(res['idx_half2'])
        stats['conf_mat_h1h2'].append(res['conf_mat_h1h2'])
        stats['conf_mat_origh1'].append(res['conf_mat_origh1'])
        stats['conf_mat_origh2'].append(res['conf_mat_origh2'])

    # Average confusion matrix across all iterations for visualization
    avg_conf_mat = np.mean(all_conf_mats, axis=0)
    
    # Store the last iteration labels for potential debugging
    half_labels = {
        'labels1': [r['labels_h1'] for r in results],
        'labels2': [r['labels_h2'] for r in results]
    }
    
    print(f"Analysis complete for K={best_k}.")
    return stats, half_labels



def plot_split_half_reliability(stats, best_k, cmap, output_path,prefix):
    """
    Visualize stability analysis results for a specific K.
    """
    
    # 1. Data preparation for Raincloud plot
    dice_data = np.array(stats['per_cluster_dice'])
    cluster_names = [f'C{i+1}' for i in range(best_k)]
    df_dice = pd.DataFrame(dice_data, columns=cluster_names)
    df_melted = df_dice.melt(var_name='Cluster', value_name='Dice Score')
    
    colors = [cmap(i) for i in range(best_k)]
    
    # 2. Initialize figure with 2x2 subplots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12)) 

    axes = axes.flatten()
    
    # --- Panel A: Heatmap 1 ---
    cm_h1h2 = np.mean(np.array(stats['conf_mat_h1h2']), axis=0)
    cm_h1h2_norm = cm_h1h2.astype('float') / (cm_h1h2.sum(axis=1)[:, np.newaxis] + 1e-10)

    sns.heatmap(cm_h1h2_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[0],
                xticklabels=cluster_names, yticklabels=cluster_names,
                square=True, cbar_kws={'label': 'Proportion of Samples'})
    
    axes[0].set_title(f"Panel A: Alignment Accuracy (K={best_k})\n(Half 1 vs. Aligned Half 2)", 
                      fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel("Aligned Half 2 Labels", fontsize=12)
    axes[0].set_ylabel("Half 1 Reference Labels", fontsize=12)

    # --- Panel B: Raincloud Plot 1 ---
    ax_b = axes[1]
    sns.violinplot(x='Cluster', y='Dice Score', data=df_melted, palette=colors, 
                   alpha=0.3, inner=None, density_norm='width', ax=ax_b, hue='Cluster', legend=False)
    
    for violin in [c for c in ax_b.collections if isinstance(c, plt.matplotlib.collections.PolyCollection)]:
        for path in violin.get_paths():
            m = path.vertices[:, 0].mean()
            path.vertices[:, 0] = np.clip(path.vertices[:, 0], m, np.inf)

    for i, cluster in enumerate(cluster_names):
        cluster_data = df_melted[df_melted['Cluster'] == cluster]
        sns.boxplot(x='Cluster', y='Dice Score', data=cluster_data, width=0.12,
                    boxprops={'facecolor': colors[i], 'edgecolor': '0.2', 'alpha': 0.8, 'zorder': 10},
                    whiskerprops={'color': '0.3', 'linewidth': 1.5},
                    capprops={'color': '0.3', 'linewidth': 1.5},
                    medianprops={'color': 'black', 'linewidth': 2.5, 'zorder': 11}, 
                    showfliers=False, ax=ax_b)

    sns.stripplot(x='Cluster', y='Dice Score', data=df_melted, palette=colors, 
                  size=3, jitter=0.15, alpha=0.5, dodge=False, ax=ax_b, hue='Cluster', legend=False)
    
    ax_b.set_title(f'Panel B: {best_k} Communities Stability\n(Dice Distribution)', fontsize=14, fontweight='bold', pad=15)
    ax_b.set_ylim(0, 1.05)
    sns.despine(ax=ax_b, left=True, bottom=True)

    # --- Panel C: Heatmap 2 ---
    cm_orighalf_data = np.concatenate((stats['conf_mat_origh1'], stats['conf_mat_origh2']))
    cm_orighalf = np.mean(cm_orighalf_data, axis=0)
    cm_orighalf_norm = cm_orighalf.astype('float') / (cm_orighalf.sum(axis=1)[:, np.newaxis] + 1e-10)

    sns.heatmap(cm_orighalf_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[2],
                xticklabels=cluster_names, yticklabels=cluster_names,
                square=True, cbar_kws={'label': 'Proportion of Samples'})
    
    axes[2].set_title(f"Panel C: Alignment Accuracy (K={best_k})\n(Origin vs. Aligned Half)", 
                      fontsize=14, fontweight='bold', pad=15)
    axes[2].set_xlabel("Aligned Half Labels", fontsize=12)
    axes[2].set_ylabel("Origin Labels", fontsize=12)

    # --- Panel D: Raincloud Plot 2 ---
    ax_d = axes[3]

    sns.violinplot(x='Cluster', y='Dice Score', data=df_melted, palette=colors, 
                   alpha=0.3, inner=None, density_norm='width', ax=ax_d, hue='Cluster', legend=False)
    
    for violin in [c for c in ax_d.collections if isinstance(c, plt.matplotlib.collections.PolyCollection)]:
        for path in violin.get_paths():
            m = path.vertices[:, 0].mean()
            path.vertices[:, 0] = np.clip(path.vertices[:, 0], m, np.inf)

    for i, cluster in enumerate(cluster_names):
        cluster_data = df_melted[df_melted['Cluster'] == cluster]
        sns.boxplot(x='Cluster', y='Dice Score', data=cluster_data, width=0.12,
                    boxprops={'facecolor': colors[i], 'edgecolor': '0.2', 'alpha': 0.8, 'zorder': 10},
                    showfliers=False,ax=ax_d)

    sns.stripplot(x='Cluster', y='Dice Score', data=df_melted, palette=colors, size=3, ax=ax_d, hue='Cluster', legend=False)
    
    ax_d.set_title(f'Panel D: {best_k} Communities Stability', fontsize=14, fontweight='bold', pad=15)
    ax_d.set_ylim(0, 1.05)
    sns.despine(ax=ax_d, left=True, bottom=True)

    # 7. Save and show
    plt.tight_layout()
    if output_path:
        os.makedirs(output_path, exist_ok=True)
        save_path = os.path.join(output_path, 'validation', f'split-half_reliability_k{best_k}_{prefix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
from matplotlib.collections import PolyCollection
def draw_raincloud_plot(ax, 
                        data, 
                        labels, 
                        colors, 
                        yrange,
                        title="", 
                        ylabel="Value", 
                        xlabel="Group"):
    """
    Internal utility to draw a horizontal or vertical raincloud plot on a given axis.
    
    Parameters:
    - ax (matplotlib.axes.Axes): The target axis to draw on.
    - data (pd.DataFrame): Long-format dataframe with columns matched to seaborn mapping.
    - labels (list): List of group names (e.g., ['C1', 'C2', ...]).
    - colors (list): List of RGB/RGBA tuples for each group.
    - title (str): Title of the subplot.
    - ylabel (str): Label for the y-axis (usually the metric name).
    - xlabel (str): Label for the x-axis (usually the group ID).
    """
    from matplotlib.ticker import MaxNLocator

    
    # 1. Draw half-violin plot (The Cloud)
    # 'density_norm' is the parameter in newer seaborn versions, replaces 'scale'
    sns.violinplot(
        x='Group', y='Score', data=data,
        palette=colors, alpha=0.3, inner=None, density_norm='width',
        ax=ax, hue='Group', legend=False
    )
    
    # Clip the violin plots to show only the right half
    for violin in [c for c in ax.collections if isinstance(c, PolyCollection)]:
        for path in violin.get_paths():
            m = path.vertices[:, 0].mean()
            path.vertices[:, 0] = np.clip(path.vertices[:, 0], m, np.inf)

    # 2. Draw boxplots (The Core)
    # We plot each box individually to ensure colors and z-order are correct
    for i, label in enumerate(labels):
        group_subset = data[data['Group'] == label]
        sns.boxplot(
            x='Group', y='Score', data=group_subset,
            width=0.12,
            boxprops={'facecolor': colors[i], 'edgecolor': '0.3', 'alpha': 0.8, 'zorder': 10},
            whiskerprops={'color': '0.3', 'linewidth': 1.5},
            capprops={'color': '0.3', 'linewidth': 1.5},
            medianprops={'color': 'black', 'linewidth': 2.0, 'zorder': 11}, 
            showfliers=False, ax=ax
        )

    # 3. Draw strip plot (The Raindrops)
    sns.stripplot(
        x='Group', y='Score', data=data,
        palette=colors, size=3, jitter=0.15, alpha=0.4,
        dodge=False, ax=ax, hue='Group', legend=False, zorder=1
    )

    # 4. Aesthetic refinements
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_ylim(yrange) 
    # ax.set_yticks(np.linspace(0, 1, 5))
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    sns.despine(ax=ax, left=True, bottom=False)
    ax.spines['bottom'].set_linewidth(0.5) 
    ax.spines['bottom'].set_color('#444444')
    return ax

def process_iteration_half_split(n_iter, half_type, labels, idx_half, base_coupling, 
                      ci_matrix_null_dicts, ci_matrix_null, labels_dict, 
                      output_path, N):
    
    network_file = f'{output_path}/half_split/network_result_{half_type}_n{n_iter}.pkl'
    coupling_file = f'{output_path}/half_split/LI_coupling_{half_type}_n{n_iter}.pkl'
    edge_file = f'{output_path}/half_split/LI_edge_{half_type}_n{n_iter}.pkl'
    

    if os.path.exists(network_file) and os.path.exists(coupling_file) and os.path.exists(edge_file):

        network_result = load_as_pickle(network_file)
        LI_coupling = load_as_pickle(coupling_file)
        LI_edge_combined = load_as_pickle(edge_file) 
        LI_obs_iter, LI_sig_iter = LI_edge_combined['obs'], LI_edge_combined['sig']
    else:
        ci_matrix = flat_to_assemble_matrix(labels[n_iter])
        
        arousal_tvFC_coupling = base_coupling[idx_half[n_iter], :]
        arousal_coupling_matrix = np.stack([
            flat_to_assemble_matrix(arousal_tvFC_coupling[i]) 
            for i in range(arousal_tvFC_coupling.shape[0])
        ])
        
        ci_matrix_dicts = {
            'cortex': ci_matrix,
            'LL': ci_matrix[:N//2, :N//2],
            'RR': ci_matrix[N//2:, N//2:],
            'LR': ci_matrix[:N//2, N//2:],
            'RL': ci_matrix[N//2:, :N//2]
        }
        
        # 1. Edge pipeline
        LI_obs_iter, LI_sig_iter = run_community_edge_part_pipeline(
            ci_matrix_dicts, ci_matrix_null_dicts, labels_dict, output_path
        )
        save_as_pickle({'obs': LI_obs_iter, 'sig': LI_sig_iter}, edge_file)
        
        # 2. Node pipeline
        network_result = run_community_node_part_pipeline(
            ci_matrix, ci_matrix_null, labels_dict, output_path, n_perm=10000
        )
        save_as_pickle(network_result, network_file)
        
        # 3. Coupling pipeline
        LI_coupling = run_community_coupling_part_pipeline(
            ci_matrix_dicts, arousal_coupling_matrix, labels_dict
        )
        save_as_pickle(LI_coupling, coupling_file)

    return {
        'n_iter': n_iter,
        'obs_edge': LI_obs_iter,
        'sig_edge': LI_sig_iter,
        'node': network_result,
        'coupling': LI_coupling
    }


def concordance_correlation_coefficient(y_true, y_pred):
    """
    Computes the Concordance Correlation Coefficient (CCC).
    CCC measures both precision (correlation) and accuracy (deviation from 45-degree line).
    Range: [-1, 1], where 1 is perfect agreement.
    """
    # Remove NaNs
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    if not np.any(mask):
        return np.nan
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    if len(y_t) < 2:
        return np.nan

    # Calculate means and variances
    mean_t = np.mean(y_t)
    mean_p = np.mean(y_p)
    var_t = np.var(y_t)
    var_p = np.var(y_p)
    covar = np.mean((y_t - mean_t) * (y_p - mean_p))
    
    # CCC Formula: (2 * covariance) / (var_t + var_p + (mean_t - mean_p)^2)
    numerator = 2 * covar
    denominator = var_t + var_p + (mean_t - mean_p)**2
    
    if denominator == 0:
        return np.nan
    return numerator / denominator

def analyze_stability_core_v1(LI_obs, 
                              LI_obs_h1_iters, 
                              LI_obs_h2_iters, 
                              min_denominator=1e-4):
    """
    Analyzes the consistency and reliability of Lateralization Index (LI) 
    using split-half iterations and Concordance Correlation Coefficient (CCC).
    
    Args:
        LI_obs: Dict containing original observed LI ('inte', 'segre').
        LI_obs_h1_iters: Dict containing LI from the first half iterations.
        LI_obs_h2_iters: Dict containing LI from the second half iterations.
        min_denominator: Small value to avoid division by zero.
    """
    n_comm = LI_obs['inte'].shape[0]
    n_net = LI_obs['inte'].shape[1]
    n_iter = len(LI_obs_h1_iters['inte'])
    types = ['inte', 'segre']
    
    # Initialize results structure
    stats_results = {t: {
        # 1. Bias metrics (Point-wise accuracy)
        'abs_bias': np.zeros((n_comm, n_net, n_net)),
        'rel_bias': np.full((n_comm, n_net, n_net), np.nan),
        'z_score': np.full((n_comm, n_net, n_net), np.nan),
        
        
        # 2. Pattern consistency (CCC is better for continuous agreement)
        'ccc_h1_vs_h2': np.full((n_comm, n_iter), np.nan), # Reliability
        'ccc_h1_vs_orig': np.full((n_comm, n_iter), np.nan), # Representativeness
        
        # 3. Summary statistics
        'avg_reliability_ccc': np.full(n_comm, np.nan),
        'avg_stability_ccc': np.full(n_comm, np.nan)
    } for t in types}
    
    for t in types:
        # Convert list to array for vectorized access
        h1_stack = np.array(LI_obs_h1_iters[t])
        h2_stack = np.array(LI_obs_h2_iters[t])
        
        for icom in range(n_comm):
            orig_pattern = LI_obs[t][icom].flatten()
            
            h1_h2_ccc_list = []
            h_orig_ccc_list = []
            
            for i in range(n_iter):
                p1 = h1_stack[i, icom].flatten()
                p2 = h2_stack[i, icom].flatten()
                
                # --- Step A: Calculate Split-half Reliability (CCC) ---
                # CCC is more strict than correlation as it checks for y=x
                c_val = concordance_correlation_coefficient(p1, p2)
                stats_results[t]['ccc_h1_vs_h2'][icom, i] = c_val
                if not np.isnan(c_val):
                    h1_h2_ccc_list.append(c_val)
                
                # --- Step B: Calculate Representativeness (CCC) ---
                c_stab = concordance_correlation_coefficient(p1, orig_pattern)
                stats_results[t]['ccc_h1_vs_orig'][icom, i] = c_stab
                if not np.isnan(c_stab):
                    h_orig_ccc_list.append(c_stab)

            # --- Step C: Aggregate CCC Scores ---
            if h1_h2_ccc_list:
                stats_results[t]['avg_reliability_ccc'][icom] = np.mean(h1_h2_ccc_list)
            
            if h_orig_ccc_list:
                stats_results[t]['avg_stability_ccc'][icom] = np.mean(h_orig_ccc_list)

            # --- Step D: Point-wise Bias and Z-score ---
            # Combine h1 and h2 to get a distribution for each element
            combined_iters = np.concatenate([h1_stack[:, icom, :, :], h2_stack[:, icom, :, :]], axis=0)
            
            m_iter = np.nanmean(combined_iters, axis=0)
            s_iter = np.nanstd(combined_iters, axis=0)
            orig_vals = LI_obs[t][icom]
            
            # 1. Absolute Bias
            abs_bias = m_iter - orig_vals
            stats_results[t]['abs_bias'][icom] = abs_bias
            
            # 2. Relative Bias
            denom = np.abs(orig_vals)
            valid_rel = denom > min_denominator
            stats_results[t]['rel_bias'][icom][valid_rel] = abs_bias[valid_rel] / denom[valid_rel]
            
            # 3. Z-Score (How many SDs original value is from iteration mean)
            valid_z = s_iter > min_denominator
            stats_results[t]['z_score'][icom][valid_z] = (orig_vals[valid_z] - m_iter[valid_z]) / s_iter[valid_z]

    return stats_results

def analyze_stability_core(obs_data, h1_stack, h2_stack, n_iter, min_denominator=1e-4):

    # 1. Pattern consistency (CCC)
    ccc_h1_h2 = np.array([concordance_correlation_coefficient(h1_stack[i].flatten(), h2_stack[i].flatten()) for i in range(n_iter)])
    ccc_h1_orig = np.array([concordance_correlation_coefficient(h1_stack[i].flatten(), obs_data.flatten()) for i in range(n_iter)])
    
    # 2. Point-wise metrics
    combined = np.concatenate([h1_stack, h2_stack], axis=0)
    m_iter = np.nanmean(combined, axis=0)
    s_iter = np.nanstd(combined, axis=0)
    
    abs_bias = m_iter - obs_data
    
    # Relative Bias
    rel_bias = np.full(obs_data.shape, np.nan)
    denom = np.abs(obs_data)
    mask_rel = denom > min_denominator
    rel_bias[mask_rel] = abs_bias[mask_rel] / denom[mask_rel]
    
    # Z-score
    z_score = np.full(obs_data.shape, np.nan)
    mask_z = s_iter > min_denominator
    z_score[mask_z] = (obs_data[mask_z] - m_iter[mask_z]) / s_iter[mask_z]
    
    return {
        'ccc_h1_vs_h2': ccc_h1_h2,
        'ccc_h1_vs_orig': ccc_h1_orig,
        'avg_reliability_ccc': np.nanmean(ccc_h1_h2),
        'avg_stability_ccc': np.nanmean(ccc_h1_orig),
        'abs_bias': abs_bias,
        'rel_bias': rel_bias,
        'z_score': z_score
    }


def analyze_LI_all_levels_stability(
    LI_obs_edge, LI_h1_edge_iters, LI_h2_edge_iters,
    LI_obs_node, LI_h1_node_iters, LI_h2_node_iters,
    LI_obs_coupling, LI_h1_coupling_iters, LI_h2_coupling_iters,
    min_denominator=1e-4
):
    
    n_iter = 500#len(LI_h1_edge_iters['inte'])
    types = ['inte', 'segre']
    results = {}

    # --- 1. Edge Level ---
    results['edge'] = {t: {} for t in types}
    for t in types:
        h1_stack = np.array(LI_h1_edge_iters[t]) # (n_iter, n_comm, n_net, n_net)
        h2_stack = np.array(LI_h2_edge_iters[t])
        n_comm = LI_obs_edge[t].shape[0]
        
        comm_results = []
        for icom in range(n_comm):
            res = analyze_stability_core(LI_obs_edge[t][icom], h1_stack[:, icom], h2_stack[:, icom], n_iter, min_denominator)
            comm_results.append(res)
        
        for key in comm_results[0].keys():
            results['edge'][t][key] = np.array([r[key] for r in comm_results])

    # --- 2. Node Level ---
    results['node'] = {t: {} for t in types}
    for t in types:

        obs_node_vals = np.array(LI_obs_node[f'mean_{t}']) # (n_comm, n_nodes)
        # h1_node_vals = np.array([it['mean'] for it in LI_h1_node_iters[t]]) # (n_iter, n_comm, n_nodes)
        
        h1_node_stack = np.swapaxes(np.array([iter_list['mean'] for iter_list in LI_h1_node_iters[t]]), 0, 1)
        h2_node_stack = np.swapaxes(np.array([iter_list['mean'] for iter_list in LI_h2_node_iters[t]]), 0, 1)

        comm_node_res = []
        for icom in range(n_comm):
            res = analyze_stability_core(obs_node_vals[icom], h1_node_stack[icom], h2_node_stack[icom], n_iter, min_denominator)
            comm_node_res.append(res)
            
        for key in comm_node_res[0].keys():
            results['node'][t][key] = np.array([r[key] for r in comm_node_res])
        results['node'][t]['half_stack'] = np.concat((h1_node_stack,h2_node_stack),axis=1)
        results['node'][t]['obs_values'] = obs_node_vals
        
    # --- 3. Coupling Level ---
    feature_full = 'slope_pair_comm'
    feat_name = feature_full.split('_')[0]
    res_key = f'coupling_{feat_name}'
    results[res_key] = {t: {} for t in types}
    for t in types:    
        obs_vals = np.array([np.nanmean(LI_obs_coupling[t][feature_full][icom]) for icom in range(n_comm)])
        h1_vals = np.swapaxes(np.array([np.mean(np.array(iter_list[feat_name]),axis=1) 
                                        for iter_list in LI_h1_coupling_iters[t]]), 
                              0, 1)
        h2_vals = np.swapaxes(np.array([np.mean(np.array(iter_list[feat_name]),axis=1) 
                                        for iter_list in LI_h2_coupling_iters[t]]), 
                              0, 1)
        comm_cp_res = []
        for icom in range(n_comm):
            res = analyze_stability_core(obs_vals[icom], h1_vals[icom], h2_vals[icom], n_iter, min_denominator)
            comm_cp_res.append(res)
            
        for key in comm_cp_res[0].keys():
            results[res_key][t][key] = np.array([r[key] for r in comm_cp_res])
        
        results[res_key][t]['half_stack'] = np.concat((h1_vals,h2_vals),axis=1)    
        results[res_key][t]['obs_values'] = obs_vals       
    feature_full = 'strength_pair_comm'
    feat_name = feature_full.split('_')[0]
    res_key = f'coupling_{feat_name}'
    results[res_key] = {t: {} for t in types}
    for t in types:    
        obs_vals = np.array([np.nanmean(LI_obs_coupling[t][feature_full][icom]) for icom in range(n_comm)])
        h1_vals = np.swapaxes(np.array([np.nanmean(np.array(iter_list[feat_name]),axis=(1,2)) 
                                        for iter_list in LI_h1_coupling_iters[t]]), 
                              0, 1)
        h2_vals = np.swapaxes(np.array([np.nanmean(np.array(iter_list[feat_name]),axis=(1,2)) 
                                        for iter_list in LI_h2_coupling_iters[t]]), 
                              0, 1)
        comm_cp_res = []
        for icom in range(n_comm):
            res = analyze_stability_core(obs_vals[icom], h1_vals[icom], h2_vals[icom], n_iter, min_denominator)
            comm_cp_res.append(res)
            
        for key in comm_cp_res[0].keys():
            results[res_key][t][key] = np.array([r[key] for r in comm_cp_res])        
        results[res_key][t]['half_stack'] = np.concat((h1_vals,h2_vals),axis=1)   
        results[res_key][t]['obs_values'] = obs_vals
        
    return results



def run_single_iteration(data, sub_ids, k, original_labels):
    
    unique_subs = np.unique(sub_ids)
    n_subs = len(unique_subs)
    n_edges = data.shape[1]
    
    sub_sample_matrix = np.zeros((n_subs, n_edges))
    
    selected_run_indices = []
    
    for i, sub in enumerate(unique_subs):
        sub_indices = np.where(sub_ids == sub)[0]  
        chosen_idx = np.random.choice(sub_indices)
        selected_run_indices.append(chosen_idx)
        sub_sample_matrix[i, :] = data[chosen_idx, :]
    
    edge_features = sub_sample_matrix.T 
    
    kmeans_sub = KMeans(n_clusters=k, n_init=10, random_state=42).fit(edge_features)
    labels_sub = kmeans_sub.labels_+1

    aligned_sub = align_to_reference(labels_sub, original_labels,k)
    # sns.heatmap(flat_to_assemble_matrix(labels_sub),vmin=1,cmap=cmap_k5)
    ari = adjusted_rand_score(original_labels, aligned_sub)
    nmi = normalized_mutual_info_score(original_labels, aligned_sub)
    
    def calculate_dice(l1, l2, k):
        dices = []
        for i in range(k):
            mask1 = (l1 == (i+1))
            mask2 = (l2 == (i+1))
            intersection = np.sum(mask1 & mask2)
            sum_val = np.sum(mask1) + np.sum(mask2)
            dice = (2. * intersection) / (sum_val + 1e-10)
            dices.append(dice)
        return np.mean(dices), dices
    
    avg_dice, per_cluster_dice = calculate_dice(original_labels, aligned_sub, k)

    return {
        "ari": ari,
        "nmi": nmi,
        "mean_dice": avg_dice,
        "per_cluster_dice": per_cluster_dice,
        "aligned_labels": aligned_sub,
        "selected_run_indices": selected_run_indices  
    }

def analyze_subject_stability(data, 
                              sub_ids, 
                              original_labels, 
                              k, 
                              n_iterations=50, 
                              n_jobs=1):

    print(f"Starting Stability Analysis: k={k}, iterations={n_iterations}")

    unique_subs = np.unique(sub_ids)
    n_edges = data.shape[1]
    full_sub_matrix = np.zeros((len(unique_subs), n_edges))
    for i, sub in enumerate(unique_subs):
        full_sub_matrix[i, :] = np.mean(data[sub_ids == sub], axis=0)
    
    kmeans_strat1 = KMeans(n_clusters=k, n_init=20, random_state=42).fit(full_sub_matrix.T)
    strat1_aligned = align_to_reference(kmeans_strat1.labels_+1, original_labels,k)
    
    ari_strat1 = adjusted_rand_score(original_labels, strat1_aligned)
    nmi_strat1 = normalized_mutual_info_score(original_labels, strat1_aligned)
    
    def calculate_dice(l1, l2, k):
        dices = []
        for i in range(k):
            mask1 = (l1 == (i+1))
            mask2 = (l2 == (i+1))
            intersection = np.sum(mask1 & mask2)
            sum_val = np.sum(mask1) + np.sum(mask2)
            dice = (2. * intersection) / (sum_val + 1e-10)
            dices.append(dice)
        return np.mean(dices), dices

    avg_dice, per_cluster_dice = calculate_dice(original_labels, strat1_aligned, k)
    
    conf_mat = confusion_matrix(
        original_labels, 
        strat1_aligned, 
        labels=range(1,k+1), 
        normalize='true'
    )
    
    print(f"Running Strategy 2 on {n_jobs} CPU cores...")
    
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(run_single_iteration)(data, sub_ids, k, original_labels) 
        for _ in range(n_iterations)
    )
    
    all_edge_labels = [r["aligned_labels"] for r in results_list]
    all_sampled_indices = [r["selected_run_indices"] for r in results_list]
    ari_list = [r["ari"] for r in results_list]
    nmi_list = [r["nmi"] for r in results_list]
    mean_dice_list = [r["mean_dice"] for r in results_list]
    all_per_cluster_dice = np.array([r["per_cluster_dice"] for r in results_list])
    
    ari_array = np.array(ari_list)

    print("Stability analysis workflow completed.")
    
    return {
        "original_labels": original_labels,
        "strat1_results": {
            "ari": ari_strat1,
            "aligned_labels": strat1_aligned,
            'confusion_matrix':conf_mat  
        },
        "strat2_metrics": {
            "ari": ari_list,
            "nmi": nmi_list,
            "mean_dice": mean_dice_list,
            "per_cluster_dice": all_per_cluster_dice,
            "all_edge_labels": all_edge_labels,       
            "all_sampled_indices": all_sampled_indices 
        },
    }
def plot_LI_stability(all_levels_obs_results, 
                      all_levels_sig_results, 
                      net_yeo7,
                      cmap_k,
                      output_path,
                      mode):
    net_yeo7  = ['DMN', 'FPN', 'LIMB', 'VAN', 'DAN', 'SMN', 'VIS']
    types = ['inte', 'segre']
    n_comm = all_levels_obs_results['edge']['inte']['abs_bias'].shape[0]
    comm_labels = [f'C{i+1}' for i in range(n_comm)]
    colors_list = [cmap_k(i) for i in range(n_comm)]
    
    sns.set_context("talk")
    sns.set_style("white")
    
    result_obs_edge = all_levels_obs_results['edge']
    result_sig_edge = all_levels_sig_results['edge']
    result_node = all_levels_obs_results['node']
    result_slope = all_levels_obs_results['coupling_slope']
    result_strength = all_levels_obs_results['coupling_strength']
    
    os.makedirs(f"{output_path}/validation", exist_ok=True)

    # --- 1 & 2. Edge & Node CCC Plots ---
    for level_name, result_dict in zip(['edge', 'node'], [result_obs_edge, result_node]):
        for t in types:
            for metric in ['ccc_h1_vs_h2', 'ccc_h1_vs_orig']:
                fig, ax = plt.subplots(figsize=(10, 5))
                df_corr = pd.DataFrame(result_dict[t][metric].T, columns=comm_labels)
                df_melted = df_corr.melt(var_name='Group', value_name='Score')
                suffix = "reliability" if "h2" in metric else "stability"
                
                ax = draw_raincloud_plot(ax, df_melted, comm_labels, colors_list, (0,1),
                                       title=f"{level_name.capitalize()}_{t}_{suffix}", 
                                       ylabel="CCC", xlabel="Communities ID")
                plt.savefig(f"{output_path}/validation/{mode}_LI-{level_name}_CCC-{suffix}_{t}.png", dpi=600, bbox_inches='tight')
                plt.close()
    # --- 3. Node Level - Stability Comparison (Box + Strip) ---
    for t in types:
        iter_data = result_node[t]['half_stack'] 
        obs_vals = result_node[t].get('obs_values', None)
        node_records, obs_records, relative_records = [], [], []

        for icom in range(n_comm):
            for inode, net_name in enumerate(net_yeo7):
                
                node_iters = iter_data[icom, :, inode]
                mean_resampled = np.mean(iter_data[icom, :, inode])
                original_val = obs_vals[icom, inode]
                if original_val != 0:
                    relative_val = ((mean_resampled - original_val) / np.abs(original_val)) * 100
                else:
                    relative_val = 0
                relative_records.append({
                    'Network': net_name,
                    'Community': f'C{icom+1}',
                    'Relative Value (%)': relative_val
                })    
                for val in node_iters:
                    node_records.append({'Network': net_name, 'Community': f'C{icom+1}', 'Lateralization index': val})
                if obs_vals is not None:
                    obs_records.append({'Network': net_name, 'Community': f'C{icom+1}', 'LI_obs': obs_vals[icom, inode]})

        df_iter, df_obs = pd.DataFrame(node_records), pd.DataFrame(obs_records).groupby(['Network', 'Community']).mean().reset_index()
        plt.figure(figsize=(12, 4))
        ax = sns.boxplot(data=df_iter, x='Network', y='Lateralization index', hue='Community', palette=colors_list, showfliers=False, whis=1.5, boxprops=dict(alpha=.3))
        sns.stripplot(data=df_obs, x='Network', y='LI_obs', hue='Community', dodge=True, marker='D', size=8, edgecolor='black', linewidth=1, palette=colors_list, ax=ax)
        ax.tick_params(axis='both', which='major', direction='out', length=6, width=1.5, colors='black', bottom=True, left=True)
        if ax.get_legend() is not None: ax.get_legend().remove()
        ax.axhline(0, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
        plt.title(f"Node-level LI Distribution across Networks ({t})")
        plt.savefig(f"{output_path}/validation/{mode}_node_stability_comparison_{t}.png", dpi=600, bbox_inches='tight')
        plt.close()
        
    

    # --- 4. Slope & Strength Stability Comparison ---
    for metric_name, result_dict, yrange in zip(['slope', 'strength'], 
                                                [result_slope, result_strength],
                                                [(0.1,0.6),(-0.02,0.02)]):
        for t in types:
            iter_data = result_dict[t]['half_stack']  # (n_comm, n_iter)
            obs_vals = result_dict[t].get('obs_values', None) # (n_comm,)
    
            records = []
            relative_records = []
            
            for icom in range(n_comm):
                iters = iter_data[icom, :]
                for val in iters:
                    records.append({'Group': f'C{icom+1}', 'Score': val})
                    
                mean_resampled = np.mean(iters)
                original_val = obs_vals[icom] if obs_vals is not None else 0
                
                if original_val != 0:
                    relative_val = ((mean_resampled - original_val) / np.abs(original_val)) * 100
                else:
                    relative_val = 0
                
                relative_records.append({
                    'Community': f'C{icom+1}',
                    'Relative Value (%)': relative_val
                })
                
            df_iter = pd.DataFrame(records)
            df_relative = pd.DataFrame(relative_records)
            
            obs_recs = []
            if obs_vals is not None:
                for icom in range(n_comm):
                    obs_recs.append({'Group': f'C{icom+1}', 'Score': obs_vals[icom]})
            df_obs = pd.DataFrame(obs_recs)
    
            fig, ax = plt.subplots(figsize=(6, 5))                
            ax = draw_raincloud_plot(ax, df_iter, comm_labels, colors_list, yrange,
                                     title=f"{metric_name} LI Stability Distribution ({t})", 
                                     ylabel="Lateralization Index", xlabel="Communities ID")
            
            if not df_obs.empty:
                sns.stripplot(data=df_obs, x='Group', y='Score', 
                              order=comm_labels, 
                              marker='D', size=10, color='white', 
                              edgecolor='black', linewidth=1.5,
                              jitter=False,
                              ax=ax, zorder=10) 
            
            if ax.get_legend() is not None:
                ax.get_legend().remove()

            ax.axhline(0, color='black', linestyle='--', alpha=0.4, zorder=0)
            plt.tight_layout()
            plt.savefig(f"{output_path}/validation/{mode}_{metric_name}_LI_stability_comparison_{t}.png", dpi=600, bbox_inches='tight')
            # plt.close()  
            
            plt.figure(figsize=(6, 4))
            ax_rel = sns.barplot(
                data=df_relative,
                x='Community',
                y='Relative Value (%)',
                palette=colors_list,
                edgecolor='black',
                linewidth=1
            )
            ax_rel.axhline(0, color='black', linestyle='-', linewidth=1.2)
            ax_rel.tick_params(axis='both', which='major', direction='out', length=6, width=1.5)
            plt.ylim(-50, 50)
            plt.title(f"Relative Deviation: {metric_name} ({t})")
            plt.ylabel("Relative Value (%)")
            plt.xlabel("Communities ID")
            plt.tight_layout()
            plt.savefig(f"{output_path}/validation/{mode}_{metric_name}_relative_deviation_{t}.png", dpi=600, bbox_inches='tight')
            plt.close()
            
    all_stats_node_records = []
    for t in types:
        iter_data = result_node[t]['half_stack']
        obs_vals = result_node[t]['obs_values']
        for icom in range(n_comm):
            for inode, net_name in enumerate(net_yeo7):
                x_bar = np.mean(iter_data[icom, :, inode])
                x_0 = obs_vals[icom, inode]
                rel_val = ((x_bar - x_0) / np.abs(x_0)) * 100 if x_0 != 0 else 0
                all_stats_node_records.append({
                    'Metric': f'Node_LI_{t}', 
                    'Community': f'C{icom+1}', 
                    'Network': net_name,
                    'Original': x_0, 
                    'Resampled_Mean': x_bar, 
                    'Relative_Error_%': rel_val
                })
    all_stats_coupling_records = []
    for metric_name, result_dict in zip(['slope', 'strength'], [result_slope, result_strength]):
        for t in types:
            iter_data = result_dict[t]['half_stack']
            obs_vals = result_dict[t]['obs_values']
            for icom in range(n_comm):
                x_bar = np.mean(iter_data[icom, :])
                x_0 = obs_vals[icom]
                rel_val = ((x_bar - x_0) / np.abs(x_0)) * 100 if x_0 != 0 else 0
                all_stats_coupling_records.append({
                    'Metric': f'{metric_name}_{t}', 
                    'Community': f'C{icom+1}', 
                    'Network': 'N/A',
                    'Original': x_0, 
                    'Resampled_Mean': x_bar, 
                    'Relative_Error_%': rel_val
                })   
                
    df_node = pd.DataFrame(all_stats_node_records)
    df_node['Absolute_Dev'] = df_node['Resampled_Mean'] - df_node['Original']
    df_node['Network'] = pd.Categorical(
        df_node['Network'], 
        categories=['DMN', 'FPN', 'LIMB', 'VAN', 'DAN', 'SMN', 'VIS'], 
        ordered=True
    )

    for metric in df_node['Metric'].unique():
        subset = df_node[df_node['Metric'] == metric]

        pivot_abs = subset.pivot(index='Community', columns='Network', values='Absolute_Dev')
        pivot_obs = subset.pivot(index='Community', columns='Network', values='Original')
        pivot_rel = subset.pivot(index='Community', columns='Network', values='Relative_Error_%')
        
        annot_matrix = []
        for i in range(pivot_abs.shape[0]):
            row_annot = []
            for j in range(pivot_abs.shape[1]):
                a_dev = pivot_abs.iloc[i, j]
                r_err = pivot_rel.iloc[i, j]
                o_val = pivot_obs.iloc[i, j]
                
                if abs(o_val) > 0.05:
                    row_annot.append(f"$\Delta$:{a_dev:.3f}\n{r_err:.1f}%")
                else:
                    row_annot.append(f"$\Delta$:{a_dev:.3f}")
            annot_matrix.append(row_annot)
    

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            pivot_abs, 
            annot=np.array(annot_matrix), 
            fmt="", 
            cmap="RdBu_r", 
            center=0,
            annot_kws={"size": 9, "va": "center"},
            linewidths=.5,
            cbar_kws={'label': 'Absolute Deviation (Resampled Mean - Original)'}
        )
        
        plt.title(f"Stability Heatmap: {metric}\n(Relative Error only shown for |X0| > 0.05)")
        plt.savefig(f"{output_path}/validation/{mode}_Heatmap_{metric}_stability.png", dpi=600, bbox_inches='tight')
        plt.show()
        
    df_coupling = pd.DataFrame(all_stats_coupling_records)
    df_coupling['Absolute_Dev'] = df_coupling['Resampled_Mean'] - df_coupling['Original']
    target_metric_order = [
        'slope_inte', 'slope_segre', 
        'strength_inte', 'strength_segre'
    ]
    df_coupling['Metric'] = pd.Categorical(
        df_coupling['Metric'], 
        categories=target_metric_order, 
        ordered=True
    )
    pivot_abs = df_coupling.pivot(index='Community', columns='Metric', values='Absolute_Dev')
    pivot_obs = df_coupling.pivot(index='Community', columns='Metric', values='Original')
    pivot_rel = df_coupling.pivot(index='Community', columns='Metric', values='Relative_Error_%')
    
    annot_matrix = []
    for i in range(pivot_abs.shape[0]):
        row_annot = []
        for j in range(pivot_abs.shape[1]):
            a_dev = pivot_abs.iloc[i, j]
            r_err = pivot_rel.iloc[i, j]
            o_val = pivot_obs.iloc[i, j]

            if abs(o_val) > 0.05:
                row_annot.append(f"$\Delta$:{a_dev:.3f}\n{r_err:.1f}%")
            else:
                row_annot.append(f"$\Delta$:{a_dev:.3f}")
        annot_matrix.append(row_annot)
    
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(
        pivot_abs, 
        annot=np.array(annot_matrix), 
        fmt="", 
        cmap="RdBu_r", 
        center=0,
        annot_kws={"size": 9, "va": "center"},
        linewidths=1,
        linecolor='white',
        cbar_kws={'label': 'Absolute Deviation ($\Delta$)'}
    )
    
    plt.title("Stability of Arousal-Coupling Metrics across Communities", pad=20)
    plt.xlabel("Coupling Metrics")
    plt.ylabel("Communities")
    
    plt.tight_layout()
    plt.savefig(f"{output_path}/validation/{mode}_Combined_Coupling_Stability_Heatmap.png", dpi=600)
    plt.show()
    
    
def Hungarian_alignment(source_labels, target_labels):
    """
    Align source_labels to target_labels using the Hungarian algorithm (Kuhn-Munkres).
    Purpose: Maximizes the overlap (Dice/Accuracy) between two clustering results 
    to solve the label switching problem.
    """
    unique_source = np.unique(source_labels)
    unique_target = np.unique(target_labels)
    
    # Calculate contingency table (intersection counts)
    # Rows = Target, Columns = Source
    cm = confusion_matrix(target_labels, source_labels)
    
    # Linear sum assignment finds the optimal mapping by minimizing cost.
    # We use -cm because the algorithm minimizes cost, and we want to maximize overlap.
    row_ind, col_ind = linear_sum_assignment(-cm)
    
    # Create mapping dictionary: {original_source_label: aligned_target_label}
    mapping = {source_lab: target_lab for target_lab, source_lab in zip(row_ind, col_ind)}
    
    # Map the labels. Use original label as fallback if mapping is missing.
    aligned_labels = np.array([mapping.get(lab, lab) for lab in source_labels])
    return aligned_labels

def _run_single_iteration(data, sub_ids, k, original_labels):
    """
    Helper function for parallel execution:
    1. Subsamples 80% of subjects.
    2. Averages within-subject runs to keep (subject x edge) feature stability.
    3. Performs KMeans on edges.
    4. Aligns results and calculates stability metrics (ARI and Dice).
    """
    unique_subs = np.unique(sub_ids)
    # Randomly select 80% of subjects without replacement
    selected_subs = np.random.choice(unique_subs, size=int(0.8 * len(unique_subs)), replace=False)
    
    n_edges = data.shape[1]
    # Construct feature matrix: rows = selected subjects, columns = edges
    sub_avg_matrix = np.zeros((len(selected_subs), n_edges))
    
    for i, sub in enumerate(selected_subs):
        sub_mask = (sub_ids == sub)
        sub_avg_matrix[i, :] = np.mean(data[sub_mask, :], axis=0)
    
    # Transpose to cluster edges: (n_edges, n_subjects_as_features)
    edge_features = sub_avg_matrix.T 
    
    # Clustering with reasonable initialization to ensure quality
    kmeans_sub = KMeans(n_clusters=k, n_init=10, random_state=None).fit(edge_features)
    labels_sub = kmeans_sub.labels_
    
    # Re-align subsampled labels to original labels to ensure C1 corresponds to C1
    aligned_sub = Hungarian_alignment(labels_sub, original_labels)
    
    # Global stability metric
    ari = adjusted_rand_score(original_labels, aligned_sub)
    
    # Community-wise stability metrics
    unique_clusters = np.sort(np.unique(original_labels))
    per_cluster_dice = []
    for cluster_id in unique_clusters:
        intersect = np.sum((original_labels == cluster_id) & (aligned_sub == cluster_id))
        total = np.sum(original_labels == cluster_id) + np.sum(aligned_sub == cluster_id)
        dice = (2.0 * intersect / total) if total > 0 else 0.0
        per_cluster_dice.append(dice)
    
    return {
        "ari": ari,
        "mean_dice": np.mean(per_cluster_dice),
        "per_cluster_dice": per_cluster_dice, # Needed for Raincloud plot
        "aligned_labels": aligned_sub
    }

def main_validation_workflow_parallel(data, sub_ids, original_labels, k, n_iterations=50, n_jobs=-1):
    """
    Main workflow for parallelized stability analysis.
    Compares two strategies:
    Strategy 1: Full-subject averaging (Deterministic baseline).
    Strategy 2: Bootstrapped subsampling (Distribution of stability).
    """
    print(f"Starting Stability Analysis: k={k}, iterations={n_iterations}")
    
    # --- Strategy 1: All-subject Averaging ---
    unique_subs = np.unique(sub_ids)
    n_edges = data.shape[1]
    full_sub_matrix = np.zeros((len(unique_subs), n_edges))
    for i, sub in enumerate(unique_subs):
        full_sub_matrix[i, :] = np.mean(data[sub_ids == sub], axis=0)
    
    # Cluster on full dataset
    kmeans_strat1 = KMeans(n_clusters=k, n_init=20, random_state=42).fit(full_sub_matrix.T)
    strat1_aligned = Hungarian_alignment(kmeans_strat1.labels_, original_labels)
    
    # Calculate Strategy 1 metrics
    ari_strat1 = adjusted_rand_score(original_labels, strat1_aligned)
    dice_strat1 = []
    for cid in np.unique(original_labels):
        it = np.sum((original_labels == cid) & (strat1_aligned == cid))
        tt = np.sum(original_labels == cid) + np.sum(strat1_aligned == cid)
        dice_strat1.append(2.0 * it / tt if tt > 0 else 0.0)
    
    # --- Strategy 2: Parallelized Subsampling ---
    print(f"Running Strategy 2 on {n_jobs if n_jobs != -1 else 'all'} CPU cores...")
    results_list = Parallel(n_jobs=n_jobs)(
        delayed(_run_single_iteration)(data, sub_ids, k, original_labels) 
        for _ in range(n_iterations)
    )
    
    # Aggregate results for distribution analysis
    ari_list = [r["ari"] for r in results_list]
    mean_dice_list = [r["mean_dice"] for r in results_list]
    
    # Extract (n_iterations, n_clusters) matrix for Raincloud plots
    all_per_cluster_dice = np.array([r["per_cluster_dice"] for r in results_list])
    
    # Find a 'typical' run (the one closest to median ARI) for visualization
    ari_array = np.array(ari_list)
    median_val = np.median(ari_array)
    median_idx = np.argmin(np.abs(ari_array - median_val))
    typical_labels = results_list[median_idx]["aligned_labels"]
    
    print("Stability analysis workflow completed.")
    
    return {
        "original_labels": original_labels,
        "strat1_results": {
            "ari": ari_strat1,
            "mean_dice": np.mean(dice_strat1),
            "per_cluster_dice": dice_strat1,
            "aligned_labels": strat1_aligned
        },
        "strat2_metrics": {
            "ari": ari_list,
            "mean_dice": mean_dice_list,
            "per_cluster_dice": all_per_cluster_dice # Array for visualization
        },
        "strat2_typical_labels": typical_labels
    }

def plot_subject_stability_metrics(result_subject, cmap, output_path):
    """
    Visualize stability analysis results:
    Panel A: Confusion Matrix for Strategy 1 (Subject Averaging vs. Original)
    Panel B: Raincloud Plot for Strategy 2 (Per-cluster Dice across iterations)
    """
    
    # 1. Extract data from results dictionary
    orig_labels = result_subject["original_labels"]
    strat1_aligned = result_subject["strat2_metrics"]["all_edge_labels"]
    
    # Expecting per_cluster_dice shape: (n_iterations, n_clusters)
    # e.g., (500, 7) for 7 communities across 500 subsampling iterations
    per_cluster_dice = result_subject["strat2_metrics"]["per_cluster_dice"] 
    n_clusters = per_cluster_dice.shape[1]
    n_iterations = per_cluster_dice.shape[0]
    
    # 2. Initialize figure with two subplots
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    # --- Panel A: Normalized Confusion Matrix (Strategy 1) ---
    # Rows represent original community size normalization
    cm_norm = np.zeros((7,7,500))
    for i in range(len(strat1_aligned)):
        cm = confusion_matrix(orig_labels, strat1_aligned[i])
        cm_norm[:,:,i] = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.mean(cm_norm,axis=2) 
    cluster_ticks = [f"C{i}" for i in np.unique(orig_labels)]
    
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=axes[0],
                xticklabels=cluster_ticks, 
                yticklabels=cluster_ticks,
                cbar_kws={'label': 'Proportion of Samples'})
    
    axes[0].set_title("Panel A: Strategy 1 Alignment Accuracy\n(Subject Averaging vs. Original)", 
                      fontsize=12, fontweight='bold', pad=15)
    axes[0].set_xlabel("Aligned Strategy 1 Labels", fontsize=12)
    axes[0].set_ylabel("Original Communities Labels", fontsize=12)
    
    # --- Panel B: Community-wise Raincloud Plot (Strategy 2) ---
    # Convert per-cluster dice to long-form DataFrame for Seaborn
    cluster_names = [f'C{i+1}' for i in range(n_clusters)]
    df_dice = pd.DataFrame(per_cluster_dice, columns=cluster_names)
    df_melted = df_dice.melt(var_name='Community', value_name='Dice Score')
    
    # Define color palette based on clusters
    if cmap is None:
        colors = sns.color_palette("husl", n_clusters)
    else:
        colors = [cmap(i) for i in range(n_clusters)]
        
    # 1. Draw the "Cloud" (Half-Violin)
    # inner=None to keep it clean, density_norm='width' to equalize visual weights
    v = sns.violinplot(
        x='Community', y='Dice Score', data=df_melted,
        palette=colors, alpha=0.3, inner=None, density_norm='width',
        ax=axes[1], hue='Community', legend=False
    )
    
    # Clip the violin plot to only show the right half
    for violin in [c for c in axes[1].collections if isinstance(c, plt.matplotlib.collections.PolyCollection)]:
        for path in violin.get_paths():
            m = path.vertices[:, 0].mean()
            path.vertices[:, 0] = np.clip(path.vertices[:, 0], m, np.inf)

    # 2. Draw the "Core" (Narrow Boxplot)
    # Placed on top of the violin to show statistical summary
    sns.boxplot(
        x='Community', y='Dice Score', data=df_melted,
        width=0.15, palette=colors, showfliers=False,
        boxprops={'alpha': 0.8, 'zorder': 10},
        medianprops={'color': 'black', 'linewidth': 2},
        ax=axes[1]
    )

    # 3. Draw the "Rain" (Jittered Stripplot)
    # Offset slightly to the left to avoid overlapping the boxplot/violin
    sns.stripplot(
        x='Community', y='Dice Score', data=df_melted,
        palette=colors, size=3, jitter=0.15, alpha=0.5,
        dodge=False, ax=axes[1], hue='Community', legend=False
    )
    
    # Adjusting title and labels for Panel B
    axes[1].set_title(f"Panel B: Community-wise Stability (Strategy 2)\n(Dice Distribution across {n_iterations} Iterations)", 
                      fontsize=12, fontweight='bold', pad=15)
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Dice Coefficient", fontsize=12)
    axes[1].set_xlabel("Communities ID", fontsize=12)
    axes[1].grid(axis='y', linestyle='--', alpha=0.4)
    
    # Final aesthetic touches
    sns.despine(ax=axes[1], left=True)
    plt.tight_layout()
    if output_path:
        plt.savefig(f'{output_path}/validation/sbj-level_result.png', dpi=600, bbox_inches='tight')
    plt.show()
    
def plot_stability_validation(data_dict, config, cmap=None, output_path=None):
    
    sns.set_theme(style="whitegrid")
    mode = config.get('mode', 'subject')
    k = config.get('best_k', 7)
    prefix = config.get('prefix', 'stability')
    
    n_rows, n_cols = (1, 2) if mode == 'subject' else (2, 2)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8*n_cols, 6*n_rows))
    axes = np.atleast_1d(axes).flatten()
    
    cluster_names = [f'C{i+1}' for i in range(k)]
    colors = [cmap(i) for i in range(k)] if cmap else sns.color_palette("husl", k)

    def draw_raincloud(ax, data, title):
        df_dice = pd.DataFrame(data, columns=cluster_names)
        df_melted = df_dice.melt(var_name='Communities ID', value_name='Dice Score')
        
        # Cloud: Violin
        sns.violinplot(x='Communities ID', y='Dice Score', data=df_melted, palette=colors, 
                       alpha=0.3, inner=None, density_norm='width', ax=ax, hue='Communities ID', legend=False)
        for violin in [c for c in ax.collections if isinstance(c, plt.matplotlib.collections.PolyCollection)]:
            for path in violin.get_paths():
                m = path.vertices[:, 0].mean()
                path.vertices[:, 0] = np.clip(path.vertices[:, 0], m, np.inf)
        
        # Core: Boxplot
        sns.boxplot(x='Communities ID', y='Dice Score', data=df_melted, width=0.15, palette=colors, 
                    showfliers=False, boxprops={'alpha': 0.8, 'zorder': 10},
                    medianprops={'color': 'black', 'linewidth': 2}, ax=ax)
        
        # Rain: Strip
        sns.stripplot(x='Communities ID', y='Dice Score', data=df_melted, palette=colors, 
                      size=3, jitter=0.15, alpha=0.5, dodge=False, ax=ax, hue='Communities ID', legend=False)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Communities ID", fontsize=14)
        ax.set_ylabel("Dice Score", fontsize=14)
        sns.despine(ax=ax, left=True)

    # ¶¨Òå»æÍ¼ÄÚ²¿¸¨Öúº¯Êý£ºHeatmap
    def draw_heatmap(ax, cm_data, title, ylabel="Reference Labels"):
        cm_norm = cm_data.astype('float') / (cm_data.sum(axis=1)[:, np.newaxis] + 1e-10)
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", ax=ax,
                    xticklabels=cluster_names, yticklabels=cluster_names,
                    square=True, cbar_kws={'label': 'Proportion of Samples'})
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel("Aligned Labels", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
    
    if mode == 'subject':
        # Panel A: Strategy 1 Alignment
        cm_raw = np.zeros((k, k, 500))
        orig_labels = data_dict["original_labels"]
        for i, aligned in enumerate(data_dict["strat2_metrics"]["all_edge_labels"]):
            from sklearn.metrics import confusion_matrix
            cm_raw[:,:,i] = confusion_matrix(orig_labels, aligned)
        draw_heatmap(axes[0], np.mean(cm_raw, axis=2), "Panel A: Alignment Accuracy\n(Subject Averaging vs. Original)")
        
        # Panel B: Dice Distribution
        draw_raincloud(axes[1], data_dict["strat2_metrics"]["per_cluster_dice"], f"Panel B: {k} Communities Stability")

    elif mode == 'split-half':
        # Panel A: H1 vs H2
        cm_h1h2 = np.mean(np.array(data_dict['conf_mat_h1h2']), axis=0)
        draw_heatmap(axes[0], cm_h1h2, f"Panel A: Alignment (K={k})\n(Half 1 vs. Aligned Half 2)")
        
        # Panel B: Dice
        draw_raincloud(axes[1], data_dict['per_cluster_dice'], "Panel B: Dice Distribution")
        
        # Panel C: Origin vs Half
        cm_orighalf_data = np.concatenate((data_dict['conf_mat_origh1'], data_dict['conf_mat_origh2']))
        draw_heatmap(axes[2], np.mean(cm_orighalf_data, axis=0), f"Panel C: Alignment (K={k})\n(Origin vs. Aligned Half)")
        
        # Panel D: Repeat Dice or Placeholder
        draw_raincloud(axes[3], data_dict['per_cluster_dice'], "Panel D: Stability Summary")

    # ------------------ ±£´æÂß¼­ ------------------
    plt.tight_layout()
    if output_path:
        os.makedirs(os.path.join(output_path, 'validation'), exist_ok=True)
        fname = f'{mode}_stability_k{k}_{prefix}.png'
        plt.savefig(os.path.join(output_path, 'validation', fname), dpi=600, bbox_inches='tight')
    plt.show()
