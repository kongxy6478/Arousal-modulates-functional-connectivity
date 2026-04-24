#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 20:36:24 2025

@author: kongxiangyu
"""
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pingouin as pg
import os
import nibabel as nib
from tqdm import tqdm
from analysis.network_mapping import aggregate_edge_to_network,aggregate_node_to_network
from scipy.stats import pearsonr, spearmanr
from src.utils import load_as_pickle, save_as_pickle
from visualization.plot_matrix import plot_half_heatmap
from scipy.stats import ttest_rel,ttest_1samp,ttest_ind
from visualization.plot_brain import plot_brainMap_bil, plot_brainMap
from analysis.slope import compute_stability
from my_code.edge_analysis import (calc_two_tailed_p,
                                   calc_LI_entropy,
                                   calc_modality_profiles,
                                   calc_community_network_pair_counts,
                                   calc_weighted_avg,
                                   calc_coupling_contribution,
                                   analyze_node_profiles,
                                   analyze_batch_null_node_profiles,
                                   analyze_network_alliliation_profiles,
                                   plot_comprehensive_coupling,
                                   plot_bar_by_modal,
                                   plot_consumbar_by_modal,
                                   plot_box_and_jitter,
                                   plot_joint_correlation)
from statsmodels.stats.multitest import multipletests,fdrcorrection


network_vertex = nib.load(
    '/data/disk0/kongxiangyu/result_explanation/GS_new/FCw_coupling_project/network/400Parcels_Yeo2011_7Networks.dlabel.nii').get_fdata().reshape(-1)
net_in_vertex = {'L': network_vertex[0:32492],
                 'R': network_vertex[32492:64984]}


def run_community_edge_full_pipeline(
    ci_matrix_dicts,
    ci_matrix_null_dicts,
    labels_dict, 
    output_path, 
    dataset,
    grad_cmaps,
    community_cmaps,
    community_colors,
    n_perm=10000, 
    plot_heatmap=True, 
    run_null=False
):
    """
    Compute and visualize coupling stability and network mapping per community.

    Parameters
    ----------
    ci_matrix_dicts : dict
        Community index matrices for 'LL' and 'RR'.
    edge_community_results : dict
        Results of edge-community metrics.
    edge_community_null_results : list of dict, optional
        Null distribution of edge-community results.
    labels_dict : dict
        Contains Yeo7 labels for LH/RH cortex.
    network_names_yeo7 : list
        List of Yeo7 network names.
    lh, rh : dict or nib objects
        Brain surface geometries.
    grad_cmaps : dict
        Community-specific color maps.
    custom_colors : list
        Colors for plotting.
    output_path : str
        Directory to save outputs.
    dataset : str
        Dataset name.
    compute_null : bool, default=True
        Whether to compute null distribution.
    plot : bool, default=True
        Whether to generate plots.

    Returns
    -------
    dict
        Dictionary containing all observed and null coupling results.
    """
    # 1. Basic Parameters & Metadata ---
    n_net = 7
    # Calculate number of communities (subtracting 0/noise if exists)
    n_comm = len(np.unique(ci_matrix_dicts['cortex'])) - 1
    communities_sample = [f'C{i}' for i in range(1, n_comm + 1)]
    
    print(f'n_comm = {n_comm}')
    # Define network label sets
    net_yeo7 = ['DMN', 'FPN', 'LIMB', 'VAN', 'DAN', 'SMN', 'VIS']
    net_yeo7x2 = [f'L_{n}' for n in net_yeo7] + [f'R_{n}' for n in net_yeo7]
    hetero_nets = ['DMN', 'FPN', 'LIMB', 'VAN', 'DAN']
    net_to_type = {n: ('H' if n in hetero_nets else 'U') for n in net_yeo7}

    # Pre-calculate edge distribution counts
    community_edge_counts = calc_community_network_pair_counts(
        ci_matrix_dicts['cortex'],
        np.concatenate((labels_dict['yeo7']['L_cortex'], labels_dict['yeo7']['R_cortex'])), 
        n_comm
    )

    # 2. Observe & Null Matrix Initialization ---
    ci_network_dict = {'mean': {ftype: np.zeros((n_comm, n_net, n_net)) for ftype in ['LL', 'RR', 'LR', 'RL']}}
    ci_network_dict['mean']['cortex'] = np.zeros((n_comm, n_net * 2, n_net * 2))
    
    ci_network_null_dict = {'mean': {ftype: np.zeros((n_comm, n_net, n_net, n_perm)) 
                                    for ftype in ['LL', 'RR', 'LR', 'RL']}}
    ci_assign_dict = {}
    
    # 3. Community-to-Network Mapping Loop ---
    file_path = f'{output_path}/REST_ci_network_null_dict_k{n_comm}.joblib'
    if not os.path.exists(file_path):
        for icom in range(n_comm):
            print(f"\n>>> Processing Community {icom+1}/{n_comm}")
            cmap = grad_cmaps[f'grad_{icom}']
            
            # Isolate edges for the current community
            ci_matrix_com = (ci_matrix_dicts['cortex'] == (icom + 1)).astype(int)
    
            # Aggregate observed edges to network matrices
            out = aggregate_edge_to_network(
                ci_matrix_com,
                row_labels=labels_dict['yeo7']['bil_cortex'],
                col_labels=labels_dict['yeo7']['bil_cortex'],
                network_names=net_yeo7x2,
                metrics=("mean", "std", 'num'),
                ftype='cortex'
            )
            ci_network_mean = out[0]['fc_network_mean']
            ci_network_dict['mean']['LL'][icom, :, :] = ci_network_mean[:n_net, :n_net].T
            ci_network_dict['mean']['RR'][icom, :, :] = ci_network_mean[n_net:, n_net:].T
            ci_network_dict['mean']['LR'][icom, :, :] = np.tril(ci_network_mean[:n_net, n_net:])
            ci_network_dict['mean']['RL'][icom, :, :] = np.triu(ci_network_mean[:n_net, n_net:]).T
            ci_network_dict['mean']['cortex'][icom, :, :] = ci_network_mean.T
    
            # Optional Null Model computation
            if run_null:
                ci_matrix_null_com = ci_matrix_null_dicts['cortex'].copy()
                ci_matrix_null_com[ci_matrix_null_com != (icom + 1)] = 0
                ci_matrix_null_com[ci_matrix_null_com != 0] = 1
                for n in tqdm(range(n_perm)):
                    out = aggregate_edge_to_network(
                        ci_matrix_null_com[n],
                        row_labels=labels_dict['yeo7']['bil_cortex'],
                        col_labels=labels_dict['yeo7']['bil_cortex'],
                        network_names=net_yeo7x2,
                        metrics=("mean", "std", 'num'),
                        ftype='cortex'
                    )
                    ci_network_mean_null = out[0]['fc_network_mean']
                    ci_network_null_dict['mean']['LL'][icom,:,:,n] = ci_network_mean_null[:n_net, :n_net].T
                    ci_network_null_dict['mean']['RR'][icom,:,:,n] = ci_network_mean_null[n_net:, n_net:].T
                    ci_network_null_dict['mean']['LR'][icom,:,:,n] = np.tril(ci_network_mean_null[:n_net, n_net:])
                    ci_network_null_dict['mean']['RL'][icom,:,:,n] = np.triu(ci_network_mean_null[:n_net, n_net:]).T
    
                save_as_pickle(ci_network_null_dict,f'{output_path}/REST_ci_network_null_dict_k{n_comm}.joblib')
        # Save mapping results
        save_as_pickle(ci_network_dict,f'{output_path}/REST_ci_network_dict_k{n_comm}.joblib')
    else:
        ci_network_dict = load_as_pickle(f'{output_path}/REST_ci_network_dict_k{n_comm}.joblib')
        ci_network_null_dict = load_as_pickle(f'{output_path}/REST_ci_network_null_dict_k{n_comm}.joblib')
    
    # 4. Lateralization Index (LI) Analysis ---
    types = ['inte', 'segre']
    LI4_obs = {t: np.zeros((n_comm, n_net, n_net)) for t in types}
    for t in types: LI4_obs[f'{t}_whole'] = np.zeros(n_comm)
    
    LI4_obs_sig = {t: np.zeros((n_comm, n_net, n_net)) for t in types}
    for t in types: LI4_obs_sig[f'{t}_whole'] = np.zeros(n_comm)

    LI4_null = {t: {'mean_pair': {f'community{i+1}': np.zeros((n_perm, n_net, n_net)) 
                                  for i in range(n_comm)}} for t in types}
    LI4_pvalue = {t: {'mean_pair': []} for t in types}

    key = 'mean'
    for icom in range(n_comm):
        # Observed quadrants
        RR, LL = ci_network_dict[key]['RR'][icom], ci_network_dict[key]['LL'][icom]
        LR, RL = ci_network_dict[key]['LR'][icom], ci_network_dict[key]['RL'][icom]
        
        # Formulas for LI
        LI4_obs['inte'][icom] = (LL + LR - RR - RL)
        LI4_obs['segre'][icom] = (LL - LR - RR + RL)

        if run_null:
            # Aggregate Null distributions
            for n in range(n_perm):
                n_RR, n_LL = ci_network_null_dict[key]['RR'][icom,:,:,n], ci_network_null_dict[key]['LL'][icom,:,:,n]
                n_LR, n_RL = ci_network_null_dict[key]['LR'][icom,:,:,n], ci_network_null_dict[key]['RL'][icom,:,:,n]
                LI4_null['inte']['mean_pair'][f'community{icom+1}'][n] = (n_LL + n_LR - n_RR - n_RL)
                LI4_null['segre']['mean_pair'][f'community{icom+1}'][n] = (n_LL - n_LR - n_RR + n_RL)

            # Significance testing
            for t in types:
                p_vals = calc_two_tailed_p(LI4_obs[t][icom], LI4_null[t]['mean_pair'][f'community{icom+1}'], True)
                LI4_pvalue[t]['mean_pair'].append(p_vals)
                
                sig_data = LI4_obs[t][icom].copy()
                sig_data[p_vals > 0.05] = np.nan
                LI4_obs_sig[t][icom] = sig_data
                
                # Weighted averages
                LI4_obs[f'{t}_whole'][icom] = calc_weighted_avg(LI4_obs[t][icom], community_edge_counts[icom])
                LI4_obs_sig[f'{t}_whole'][icom] = calc_weighted_avg(sig_data, community_edge_counts[icom])

    # Hemisphere-specific dominance separation (L vs R)
    for t in types:
        for suffix, target_dict in [('', LI4_obs), ('_sig', LI4_obs_sig)]:
            l_key, r_key = f'{t}_L', f'{t}_R'
            target_dict[l_key], target_dict[r_key] = target_dict[t].copy(), target_dict[t].copy()
            target_dict[l_key][target_dict[l_key] < 0] = np.nan
            target_dict[r_key][target_dict[r_key] > 0] = np.nan

    save_as_pickle(LI4_obs_sig, f'{output_path}/LI4_obs_sig_k{n_comm}.joblib')
    save_as_pickle(LI4_obs, f'{output_path}/LI4_obs_k{n_comm}.joblib')
    
    # 5. Visualization Section
    # ---- Figure 2. B top
    print("\n>>>> Start Figure 2B-top")
    for icom in range(n_comm):
        print(f"\n>>> Community {icom+1}/{n_comm} ===")
        cmap = grad_cmaps[f'grad_{icom}']
        
        plot_half_heatmap(
           pd.DataFrame(ci_network_dict['mean']['cortex'][icom, :, :],
                        index=net_yeo7x2,
                        columns=net_yeo7x2),
           vrange=[0, 0.7], cmap=cmap, half='lower',
           labels=net_yeo7x2,
           prefix=f'{dataset}_ci-net-mean-cortex_community{icom+1}_K{n_comm}',
           output_path=output_path,
           cbar_legend = 'proportion'
        )
        
    # ---- Figure 2. B
    print("\n>>>> Start Figure 2B-bottom")
    hetero_nets_2x7 = [f'{h}_{n}' for h in ['L','R'] for n in hetero_nets]
    net_to_type_2x7 = {
        net: 'H' if net in hetero_nets_2x7 else 'U' for net in net_yeo7x2}
    df_modal = calc_modality_profiles(ci_network_dict['mean']['cortex'],
                                      net_yeo7x2, 
                                      net_to_type_2x7)
    df_modal_long = pd.melt(
        df_modal, 
        id_vars=['community', 'total_edges'], # Columns to keep as identifiers
        value_vars=['U - U', 'H - H', 'U - H'], # Columns to unpivot
        var_name='connection_type',             # Name for the new category column
        value_name='proportion'                 # Name for the new value column
    )
    communities = df_modal['community'].unique()
    plot_box_and_jitter(
        df=df_modal, 
        x_col='community', 
        y_col='value', 
        hue_col='modal', 
        palette_name='PuBu',
        box_width=0.5,
        xtricks_list=communities,
        output_path=output_path,
        prefix=f'modal-weight-all-community_K{n_comm}',
        dataset_name='REST',
        title='Proportion by connection modal type across community',
        y_label='Proportion',
        x_label='Community ID',
        jitter_w=0.03,
        legend_loc='upper right',
        figsize=(12, 4),
        yrange=None,
        dpi=600
    )
    
    # ---- Figure 2. C    
    print("\n>>>> Start Figure 2C")
    comm_netmat = ci_network_dict['mean']['cortex'].copy()
    # n_comm, n_net, _ = comm_netmat.shape
    total_edges = comm_netmat.sum(axis=0, keepdims=True)   # shape = (1, n_net, n_net)
    prob = comm_netmat / (total_edges + 1e-12)                # shape = (n_comm, n_net, n_net)
    entropy_all = -np.sum(prob * np.log(prob + 1e-12), axis=0)    # shape = (n_net, n_net)
    entropy_norm = entropy_all / np.log(n_comm)
    entropy_norm[total_edges.squeeze(0) == 0] = np.nan
    ci_assign_dict['entropy'] = entropy_norm
    
    plot_half_heatmap(
            pd.DataFrame(ci_assign_dict['entropy'],index=net_yeo7x2,columns=net_yeo7x2),
            vrange=[0.5, 1], cmap='PuBu', half='lower',
            labels=net_yeo7x2,
            prefix=f'{dataset}_ci-net-entropy-cortex_K{n_comm}',
            output_path=output_path,
            cbar_legend = 'Entropy'
        )
    
    plot_bar_by_modal(matrix_data=ci_assign_dict['entropy'],
                      network_names=net_yeo7x2,
                      dataset='REST',
                      prefix=f'Entropy_K{n_comm}',
                      output_path=output_path)
    
    # ---- Figure 2. D
    print("\n>>>> Start Figure 2D")
    ci_assign_dict['max'] = np.argmax(ci_network_dict['mean']['cortex'],axis=0)+1
    plot_half_heatmap(
            pd.DataFrame(ci_assign_dict['max'],
                         index=net_yeo7x2,
                         columns=net_yeo7x2),
            vrange=[1,n_comm], cmap=community_cmaps, half='lower',
            labels=net_yeo7x2,
            prefix=f'{dataset}_ci-net-max-cortex_K{n_comm}',
            output_path=output_path,
            cbar_legend='Community ID'
        )
    plot_consumbar_by_modal(matrix_data=ci_assign_dict['max'],
                            network_names=net_yeo7x2,
                            custom_colors=community_colors,
                            dataset='REST',
                            prefix=f'Max_K{n_comm}',
                            output_path=output_path)

    
    # ---- Figure 3. A & B
    print("\n>>>> Start Figure 3 A & B")
    for icom in range(n_comm):
        print(f"\n>>> Community {icom+1}/{n_comm} ===")
        key='mean'
        for LI_type in ['segre','inte']:
            plot_half_heatmap(
                LI4_obs_sig[LI_type][icom,:,:],
                # pd.DataFrame(LI4_obs_sig[LI_type][icom,:,:],
                #              index=net_yeo7,
                #              columns=net_yeo7),
                vrange=[-0.5, 0.5], cmap='RdBu_r', half='lower',
                # labels=net_yeo7,
                prefix=f'{dataset}_ci-net-mean-{LI_type}-sig_com{icom+1}_K{n_comm}',
                output_path=output_path
            ) 
            
            plot_half_heatmap(
                LI4_obs[LI_type][icom],
                # pd.DataFrame(LI4_obs[LI_type][icom],
                #              index=net_yeo7,
                #              columns=net_yeo7),
                vrange=[-0.5, 0.5], cmap='RdBu_r', half='lower',
                # labels=net_yeo7,
                prefix=f'{dataset}_ci-net-mean-{LI_type}_com{icom+1}_K{n_comm}',
                output_path=output_path
            )    
            
    # ---- Figure 3. C & D
    print("\n>>>> Start Figure 3 C & D")
    key_titles = ['Integration', 'Segregation']
    metrics = ['inte', 'segre']
    from scipy.stats import kruskal
    for i, key in enumerate(metrics):
        # --- 1. Data Preparation ---
        # Fetch modal data for Left and Right hemispheres
        df_L = calc_modality_profiles(LI4_obs[f'{key}_L'], net_yeo7, net_to_type)
        df_R = calc_modality_profiles(LI4_obs[f'{key}_R'], net_yeo7, net_to_type)
        
        # Annotate hemisphere and update modal labels for unique identification
        df_L["hemisphere"], df_R["hemisphere"] = "L", "R"
        df_L['modal'] = df_L['modal'] + f' - {key} - L'
        df_R['modal'] = df_R['modal'] + f' - {key} - R'
        
        # Combine L and R data
        df_combined = pd.concat([df_L, df_R], ignore_index=True)
        
        # Calculate across-community mean for each modal type
        df_mean = df_combined.groupby("modal")["value"].mean().reset_index()
        df_mean["community"] = "mean"
        df_combined = pd.concat([df_combined, df_mean], ignore_index=True)
        
        # Define categorical order for communities (ensure 'mean' is included)
        comm_categories = list(dict.fromkeys(df_combined["community"].tolist()))
        df_combined["community"] = pd.Categorical(df_combined["community"], 
                                                  categories=comm_categories, 
                                                  ordered=True)
        
        # --- 2. Color Palette Configuration ---
        # Define warm colors for Left and cool colors for Right
        colors_L = sns.color_palette("OrRd", 3)   
        colors_R = sns.color_palette("PuBu", 3)
        
        unique_modals = df_combined['modal'].unique()
        palette = {}
        idx_L, idx_R = 0, 0
        for m in unique_modals:
            if f'{key} - L' in m:
                palette[m] = colors_L[idx_L]
                idx_L += 1
            else:
                palette[m] = colors_R[idx_R]
                idx_R += 1
    
        # --- 3. Visualization ---
        plt.figure(figsize=(16, 4))
        # Barplot for mean values
        ax = sns.barplot(
            data=df_combined, x='community', y='value', 
            hue='modal', palette=palette
        )
        # Stripplot for individual data points
        sns.stripplot(
            data=df_combined, x='community', y='value', 
            hue='modal', palette=palette,
            dodge=True, jitter=True, alpha=1, 
            linewidth=0.4, edgecolor='black', legend=False
        )
        # Plot styling
        plt.ylim((-0.4, 0.4))
        plt.title(f'Mean {key_titles[i]} by Modal Type', fontsize=18)
        plt.ylabel(f'Mean {key_titles[i]}', fontsize=18)
        plt.xlabel('Community ID', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=18)
        # Legend handling: only show legend for the first plot (Integration)
        if key == 'inte':
            ax.legend(title='Modal Type', fontsize=14, title_fontsize=12,
                      loc='lower center', bbox_to_anchor=(0.5, 1.1), ncol=3)
        else:
            ax.get_legend().remove()    
        # Save figures
        for fmt in ['png', 'eps']:
            save_path = f'{output_path}/{fmt}/{dataset}_mean_{key}-modal-all_community_K{n_comm}.{fmt}'
            plt.savefig(save_path, dpi=600, bbox_inches='tight')
        plt.show()
        
        # --- 4. Statistical Testing (Kruskal-Wallis) ---
        print(f"\n{'='*20} Stats: {key_titles[i]} {'='*20}")
        for hemi in ["L", "R"]:
            # Filter data: specific hemisphere and exclude the summary 'mean' column
            df_stat = df_combined[(df_combined["hemisphere"] == hemi) & (df_combined["community"] != "mean")]
            groups = [g["value"].values for _, g in df_stat.groupby("modal")]
            stat, p = kruskal(*groups)
            print(f"Hemisphere {hemi}: H-statistic = {stat:.4f}, p-value = {p:.4e}")
                
        # ---- Figure 3. E
        print("\n>>>> Start Figure 3E")
        inte_entropy = calc_LI_entropy(LI4_obs, community_edge_counts, 'inte')
        segre_entropy = calc_LI_entropy(LI4_obs, community_edge_counts, 'segre')
        df_metrics = pd.DataFrame({
            'inte_whole': [LI4_obs['inte_whole'][i] for i in range(n_comm)],
            'segre_whole': [LI4_obs['segre_whole'][i] for i in range(n_comm)],
            'inte_entropy': inte_entropy,
            'segre_entropy': segre_entropy,
        }, index=[f'C{i+1}' for i in range(n_comm)])
    
        fig, ax = plt.subplots(figsize=(6, 6))
        for i, com in enumerate(df_metrics.index):
            x = df_metrics['inte_whole'][i]
            y = df_metrics['segre_whole'][i]
            
            ax.scatter(x, y, color='steelblue', s=250, edgecolors='white', zorder=5)
            ax.text(x-0.02, y + 0.05, com, fontsize=16, ha='center', va='bottom')
    
        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.axvline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlim([-0.3, 0.3]) 
        ax.set_ylim([-0.3, 0.3])
        ax.set_xlabel('Integration', fontsize=16)
        ax.set_ylabel('Segregation', fontsize=16)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        ax.set_title('Coordination of integration and segregation', fontsize=16,pad=15)
    
        plt.tight_layout()
        plt.savefig(f"{output_path}/png/{dataset}_LI4_coordinate_K{n_comm}.png", dpi=600)
        plt.savefig(f"{output_path}/eps/{dataset}_LI4_coordinate_K{n_comm}.eps", dpi=600)
        plt.show()
    
    # ---- Figure 3. F 
    print("\n>>>> Start Figure 3F")
    tril_net_index = np.tril_indices(7)
    for i, key in enumerate(['inte','segre']):
        community_key = np.stack([LI4_obs[key][i][tril_net_index] for i in range(n_comm)])
        rho,p_rho = spearmanr(community_key.T,nan_policy='omit')
        plt.figure(figsize=(8, 6))
        g=sns.clustermap(pd.DataFrame(data=rho,
                                 index=communities_sample,
                                 columns=communities_sample),
                    cmap='RdBu_r',square=True,vmax=1,vmin=-1,figsize=(4,4),
                    cbar=False,cbar_kws={'label': ''})
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=14, rotation=0)
        g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=14, rotation=0)
        g.ax_heatmap.tick_params(axis='both', which='major', length=0)
        g.ax_cbar.set_ylabel('')
        g.ax_cbar.set_xticks([])
        g.ax_cbar.set_yticks([])
        g.fig.savefig(f'{output_path}/png/heatmap_spearman_community_{key}_K{n_comm}.png',dpi=600)
        g.fig.savefig(f'{output_path}/eps/heatmap_spearman_community_{key}_K{n_comm}.eps',dpi=600)
        plt.show()
    
def run_community_node_full_pipeline(ci_matrix, 
                                     ci_matrix_null, 
                                     labels_dict,
                                     lh, rh, 
                                     output_path,
                                     dataset, 
                                     grad_cmaps, 
                                     community_colors,
                                     community_cmaps, 
                                     n_perm=10000,
                                     plot_brain=True, 
                                     plot_corr=True):
    """
    Full pipeline for edge community metrics analysis.

    Steps
    -----
    1. Compute observed and null metrics
    2. Plot brain maps for observed metrics
    3. Save node affiliations (observed & null)
    4. Compute L-R node entropy correlation
    5. Test L-R significance using null

    Parameters
    ----------
    ci_matrix : ndarray
        Observed community assignments (n_edges ¡Á n_states)
    ci_matrix_null : list of ndarray
        Null community assignments (permutation ¡Á n_edges ¡Á n_states)
    labels_dict : dict
        ROI label mapping (id -> network)
    network_names_yeo17 : list
        List of network names for visualization
    lh, rh : list
        Left and right hemisphere surface coordinates or geometry
    output_path : str
        Output directory for saving results
    dataset : str
        Dataset name (e.g., 'REST' or 'MOVIE')
    community_cmaps : Colormap
        Custom colormap for visualization
    n_perm : int, default=1000
        Number of null permutations
    plot_brain : bool, default=True
        Whether to plot brain maps for observed data
    plot_corr : bool, default=True
        Whether to plot LL¨CRR node entropy correlation
    """
   
    n_net = 7
    n_comm = len(np.unique(ci_matrix))-1
    net_yeo7 = ['DMN', 'FPN', 'LIMB', 'VAN', 'DAN','SMN', 'VIS']

    # ---- Figure 4A
    print("\n>>>> Start Figure 4A")
    print('\n>>> Step1: Compute observed metrics')
    file_path = f'{output_path}/{dataset}_community_node_results_k{n_comm}.joblib'
    if not os.path.exists(file_path):
        community_node_results = analyze_node_profiles(
            ci_matrix, 
            labels_dict, 
            net_yeo7,
            output_path, 
            prefix=f"{dataset}_edge-community",
            custom_cmap=community_cmaps
        )
        save_as_pickle(community_node_results,file_path)
    else:
        community_node_results = load_as_pickle(file_path)

    print("\n>>> Step2: Compute null metrics (Permutations)")
    file_path = f'{output_path}/{dataset}_community_node_results_null_k{n_comm}.joblib'
    if not os.path.exists(file_path):
        community_node_results_null = analyze_batch_null_node_profiles(
            ci_matrix_null,
            labels_dict, 
            net_yeo7,
            output_path, 
            prefix=f"{dataset}_edge-community",
            custom_cmap=community_cmaps, 
            n_jobs=40,
        )
        save_as_pickle(community_node_results_null,file_path)
    else:
        community_node_results_null = load_as_pickle(file_path)
    
    # ---- Figure 4B    
    print("\n>>>> Start: Figure 4B")    
    print("\n>>> Step1: Plot node affiliations")
    community_node_affi_all = {
        ftype: community_node_results[ftype]['affiliation']
        for ftype in ['LL', 'RR', 'LR', 'RL']
    }
    community_node_affi_all_null = {
        ftype: np.stack([community_node_results_null[i][ftype]['affiliation']
                         for i in range(n_perm)])
        for ftype in ['LL', 'RR', 'LR', 'RL']
    }
    save_as_pickle(community_node_affi_all,f'{output_path}/{dataset}_community_node_affi_all_k{n_comm}.joblib')
    
    if plot_brain:
        for icom in range(n_comm):
            print(f"\n>>> Plotting Brain Maps for Community {icom+1}/{n_comm}")
            cmap = grad_cmaps[f'grad_{icom}']

            # Plot LL-LR and RL-RR combinations
            plot_brainMap_bil(
                [community_node_affi_all['LL'][icom], 
                 community_node_affi_all['LR'][icom]], 
                [0, 0.4], [lh, rh], cmap=cmap, 
                prefix=f'{dataset}_node-affiliation_LLLR_com{icom+1}_K{n_comm}',
                save_path=output_path
            )
            
            plot_brainMap_bil(
                [community_node_affi_all['RL'][icom], 
                 community_node_affi_all['RR'][icom]],           
                [0, 0.4], [lh, rh], cmap=cmap,             
                prefix=f'{dataset}_node-affiliation_RLRR_com{icom+1}_K{n_comm}',
                save_path=output_path
            )
    
    # ---- Figure 4C
    print("\n>>>> Start: Figure 4C")     
    if plot_brain:
        print("\n>>> Step1: Plot observed results")
        # Additional bilateral brain maps
        plot_brainMap_bil(
            [community_node_results['LL']['node_entropy'],
             community_node_results['LR']['node_entropy']],
            [0.4, 1], [lh, rh], cmap='PuBu',
            prefix=f'{dataset}_LLLR_node_entropy_K{n_comm}',
            save_path=output_path
        )
        plot_brainMap_bil(
            [community_node_results['RL']['node_entropy'],
             community_node_results['RR']['node_entropy']],
            [0.4, 1], [lh, rh], cmap='PuBu',
            prefix=f'{dataset}_RLRR_node_entropy_K{n_comm}',
            save_path=output_path
        )

    # ---- Figure 4D
    print("\n>>>> Start: Figure 4D")  
    # Build Entropy DataFrame
    ftypes = ['LL', 'RR', 'LR', 'RL']
    entropy_vals = np.concatenate([community_node_results[f]['node_entropy'] for f in ftypes])
    ftype_labels = np.concatenate([np.repeat(f, 200) for f in ftypes])
    net_labels = np.concatenate([labels_dict['yeo7']['L_cortex'], labels_dict['yeo7']['R_cortex'], 
                                 labels_dict['yeo7']['R_cortex'], labels_dict['yeo7']['L_cortex']])
    
    df_entropy = pd.DataFrame({'value': entropy_vals, 'ftype': ftype_labels, 'network': net_labels})
    
    # Statistical comparison between Association (Part 1) and Sensorimotor (Part 2) networks
    df_entropy['network_group'] = df_entropy['network'].apply(
        lambda x: 'Part1' if x in ['DMN','FPN','LIMB'] else 
                  'Part2' if x in ['DAN','SMN','VIS','VAN'] else 'Other')
    
    df_2part = df_entropy[df_entropy['network_group'].isin(['Part1','Part2'])]
    print("T-Test (Part1 vs Part2 Entropy):")
    print(pg.ttest(df_2part[df_2part['network_group']=='Part1']['value'], 
                   df_2part[df_2part['network_group']=='Part2']['value']))
    print(ttest_ind(df_2part[df_2part['network_group']=='Part1']['value'], 
                    df_2part[df_2part['network_group']=='Part2']['value']))

    plot_box_and_jitter(
        df=df_entropy, 
        x_col='network', 
        y_col='value', 
        hue_col='ftype', 
        palette_name='PuBu',
        box_width=0.5,
        xtricks_list=net_yeo7,
        yrange=(0,1),
        output_path=output_path,
        prefix=f'network-entropy-all-community_yeo7_K{n_comm}',
        dataset_name='REST',
        title='Network-level entropy across community',
        y_label='Entropy',
        x_label='Yeo7 network',
        jitter_w=0.03,
        legend_loc='best',
        figsize=(12, 4),
        dpi=600
    )
       
    # ---- Figure 4E & F
    print("\n>>>> Start: Figure 4 E & F")  
    file_path = f'{output_path}/{dataset}_network_result_k{n_comm}.joblib'
    # if not os.path.exists(file_path):
    network_result = analyze_network_alliliation_profiles(
        community_node_affi_all,
        community_node_affi_all_null,
        labels_dict,
        net_yeo7,
        community_cmaps,
        output_path,
        dataset='REST'
    )
    save_as_pickle(network_result, file_path)
    # else:
    #     network_result = load_as_pickle(file_path)
   
    for metric, y_range, title in [('inte', (-0.3, 0.3), 'Integration'), 
                                   ('segre', (-0.8, 0.8), 'Segregation')]:
        df_plot = network_result[f'df_all_{metric}'].copy()
        # Add jitter for visualization
        df_plot["value_jit"] = df_plot["value"] + np.random.normal(0, 1e-2, len(df_plot))
        
        plot_box_and_jitter(
            df=df_plot, 
            x_col='network', 
            y_col='value_jit', 
            hue_col='community', 
            palette_name=community_colors, 
            xtricks_list=net_yeo7, 
            yrange=y_range, 
            output_path=output_path, 
            prefix=f'network-{title.lower()}-all-community_K{n_comm}', 
            dataset_name=dataset, 
            title=f'Network-level {title} per Community', 
            y_label=title,
            point_size=3,
            jitter_w=0.05,
            box_width=0.7,
            show_legend=False
        )
    def apply_fdr_correction(p_df, alpha=0.05):
        """
        ¶Ô DataFrame ¸ñÊ½µÄ P Öµ¾ØÕó½øÐÐ FDR ÐÞÕý¡£
        
        ²ÎÊý:
        - p_df: pd.DataFrame, Ô­Ê¼ P Öµ¾ØÕó
        - alpha: float, ÏÔÖøÐÔË®Æ½ (Ä¬ÈÏ 0.05)
        
        ·µ»Ø:
        - sig_mask: pd.DataFrame (bool), ÏÔÖøÐÔÑÚÂë¾ØÕó (True ±íÊ¾ÏÔÖø)
        - p_adj_df: pd.DataFrame (float), ÐÞÕýºóµÄ P Öµ¾ØÕó
        """
        n_rows, n_cols = p_df.shape
        index_names = p_df.index
        column_names = p_df.columns
    
        reject, p_adj = fdrcorrection(p_df.values.flatten(), alpha=alpha)
        
        sig_mask = pd.DataFrame(reject.reshape((n_rows, n_cols)), 
                                 index=index_names, columns=column_names)
        
        p_adj_df = pd.DataFrame(p_adj.reshape((n_rows, n_cols)), 
                                 index=index_names, columns=column_names)
        
        return sig_mask, p_adj_df
    
    sig_inte, padj_inte = apply_fdr_correction(network_result['p_inte'], alpha=0.05)

    sig_segre, padj_segre = apply_fdr_correction(network_result['p_segre'], alpha=0.05)

    # _ , p_adj = fdrcorrection(network_result['p_inte'].flatten(),0.05)
    # sig_inte = p_adj.reshape((n_comm,n_net))
    # _ , p_adj = fdrcorrection(network_result['p_segre'].flatten(),0.05)
    # sig_segre = p_adj.reshape((n_comm,n_net))
    
    # sig_inte = network_result['sig_inte']
    # sig_segre = network_result['sig_segre']
    print(sig_inte)
    print(sig_segre)
    print(padj_inte)
    print(padj_segre)
    
    # df = df_plot
    # df = network_result['df_all_segre']
    # specific_mean = df[(df['community'] == 'C3') & 
    #                (df['network'] == 'DMN')]['value_jit'].mean()
    # print(specific_mean)
    
    # null_save_data=np.load(f"{output_path}/{dataset}_network_affiliation_null_dist.npz")
    # null_segre_all = null_save_data['null_segre_all']
    # null_dist = null_segre_all[0,:,0]
    
    
def run_community_coupling_full_pipeline(
    ci_matrix_dicts,
    arousal_coupling_matrix,
    labels_dict,
    grad_cmaps, 
    custom_colors, 
    output_path, 
    dataset
):
    """
    Compute and visualize coupling stability and network mapping per community.

    Parameters
    ----------
    ci_matrix_dicts : dict
        Community index matrices for 'LL' and 'RR'.
    edge_community_results : dict
        Results of edge-community metrics.
    edge_community_null_results : list of dict, optional
        Null distribution of edge-community results.
    FC_coupling_dict : dict
        Coupling matrices per subject and hemisphere.
    labels_dict : dict
        Contains Yeo7 labels for LH/RH cortex.
    network_names_yeo7 : list
        List of Yeo7 network names.
    lh, rh : dict or nib objects
        Brain surface geometries.
    grad_cmaps : dict
        Community-specific color maps.
    custom_colors : list
        Colors for plotting.
    output_path : str
        Directory to save outputs.
    dataset : str
        Dataset name.
    compute_null : bool, default=True
        Whether to compute null distribution.
    plot : bool, default=True
        Whether to generate plots.

    Returns
    -------
    dict
        Dictionary containing all observed and null coupling results.
    """
    
    # --- 0. Parameters & Setup ---
    net_yeo7 = ['DMN', 'FPN', 'LIMB', 'VAN', 'DAN', 'SMN', 'VIS']
    net_yeo7x2 = [f'L_{n}' for n in net_yeo7] + [f'R_{n}' for n in net_yeo7]
    n_comm = len(np.unique(ci_matrix_dicts['cortex'])) - 1
    n_run, n_roi = arousal_coupling_matrix.shape[0], 200
    n_net, n_pair = 7, 28
    tril_net_index = np.tril_indices(n_net)
    triu_edge_index = np.triu_indices(n_roi)
    ftype_list = ['LL', 'RR', 'LR', 'RL']
      
    coupling_result_dict = {f'C{icom}': {} for icom in range(n_comm)}  
    communities = [f'C{i}' for i in range(1,n_comm+1)]  
    communities_wb = np.hstack((communities,'wb'))
    # coupling_dict = {'network': {ftype: np.full((n_comm, n_net, n_net), np.nan)
    #                              for ftype in ftype_list},
    #                  'edge': {ftype: np.full((n_comm, n_roi, n_roi), np.nan)
    #                           for ftype in ftype_list},
    #                  'all_network': {ftype: np.full((n_comm, n_net, n_net), np.nan)
    #                              for ftype in ftype_list}
    #                  }
    coupling_data = {m: {f: np.full((n_comm, n_net, n_net), np.nan) for f in ftype_list} 
                         for m in ['comm', 'allcomm']}
    coupling_data['edge'] = {f: np.full((n_comm, n_roi, n_roi), np.nan) for f in ftype_list}
    
    def compute_stability_batch(data_vect):
        data_vect[data_vect == 0] = np.nan
        notnan_idx = ~np.isnan(data_vect)
        if not np.any(notnan_idx): return 0, 0
        vals_sorted = np.sort(data_vect[notnan_idx])[::-1]
        return compute_stability(vals_sorted)
    
    file_path = f'{output_path}/coupling_result_dict_k{n_comm}.joblib'
    if not os.path.exists(file_path):
        for icom in range(n_comm):
            print(f"\n>>> Compute coupling in Community {icom+1}/{n_comm}")
            cmap = grad_cmaps[f'grad_{icom}']
            
            mask_com = (ci_matrix_dicts['cortex'] == (icom + 1)).astype(float)
            coupling_result = {
                # 'slope_net': {ftype: np.zeros(n_run) for ftype in ftype_list},
                # 'slope_whole':{ftype: np.zeros(n_run) for ftype in ftype_list},
                'strength_pair_allcomm': {ftype: np.zeros((n_run, n_pair)) for ftype in ftype_list},
                'strength_pair_comm': {ftype: np.zeros((n_run, n_pair)) for ftype in ftype_list},
                # 'strength_allnet': {ftype: np.zeros((n_run, n_pair)) for ftype in ftype_list},
                # 'strength_node': {ftype: np.zeros((n_run, n_roi)) for ftype in ftype_list}
            }
            for i in tqdm(range(n_run)):
                
                curr_coupling = arousal_coupling_matrix[i].copy()
                curr_coupling_com = curr_coupling * mask_com
                # curr_coupling_com = curr_coupling_com + curr_coupling_com.T
                curr_coupling_com[curr_coupling_com == 0] = np.nan
                
                out_comm = aggregate_edge_to_network(
                    curr_coupling_com, 
                    labels_dict['yeo7']['bil_cortex'],          
                    labels_dict['yeo7']['bil_cortex'], 
                    metrics=("mean", "std", 'num'),
                    network_names=net_yeo7x2, 
                    ftype='cortex')
                
                net_mean = out_comm[0]['fc_network_mean']
                coupling_data['comm']['LL'][icom] = net_mean[:n_net,:n_net].T
                coupling_data['comm']['RR'][icom] = net_mean[n_net:,n_net:].T
                coupling_data['comm']['LR'][icom] = np.tril(net_mean[:n_net,n_net:])
                coupling_data['comm']['RL'][icom] = np.triu(net_mean[:n_net,n_net:]).T
                coupling_data['comm']['LR'][icom][coupling_data['comm']['LR'][icom]==0]=np.nan
                coupling_data['comm']['RL'][icom][coupling_data['comm']['RL'][icom]==0]=np.nan
                                 
                out_all = aggregate_edge_to_network(
                    curr_coupling,
                    row_labels=labels_dict['yeo7']['bil_cortex'],
                    col_labels=labels_dict['yeo7']['bil_cortex'],
                    network_names=net_yeo7x2,
                    metrics=("mean", "std", 'num'),
                    ftype='cortex'
                )
                allcomm_mean = out_all[0]['fc_network_mean']
                coupling_data['allcomm']['LL'][icom] = allcomm_mean[:n_net,:n_net].T
                coupling_data['allcomm']['RR'][icom] = allcomm_mean[n_net:,n_net:].T
                coupling_data['allcomm']['LR'][icom] = np.tril(allcomm_mean[:n_net,n_net:])
                coupling_data['allcomm']['RL'][icom] = np.triu(allcomm_mean[:n_net,n_net:]).T
                coupling_data['allcomm']['LR'][icom][coupling_data['allcomm']['LR'][icom]==0]=np.nan
                coupling_data['allcomm']['RL'][icom][coupling_data['allcomm']['RL'][icom]==0]=np.nan
                
                # coupling_data['edge']['LL'][icom] = curr_coupling_com[:n_roi,:n_roi].T
                # coupling_data['edge']['RR'][icom] = curr_coupling_com[n_roi:,n_roi:].T
                # coupling_data['edge']['LR'][icom] = np.tril(curr_coupling_com[:n_roi,n_roi:])
                # coupling_data['edge']['RL'][icom] = np.triu(curr_coupling_com[:n_roi,n_roi:]).T
                # coupling_data['edge']['LR'][icom][coupling_data['edge']['LR'][icom]==0]=np.nan
                # coupling_data['edge']['RL'][icom][coupling_data['edge']['RL'][icom]==0]=np.nan
                
                for f in ftype_list:
                    for key in ['comm', 'allcomm']:
                        # Extract vector using indices
                        vec = coupling_data[key][f][icom][tril_net_index]
                        coupling_result[f'strength_pair_{key}'][f][i] = vec
                
                    # # 2. Node-level simplification
                    # edge_data = coupling_data['edge'][f][icom]
                    # coupling_result['strength_node'][f][i] = np.nan_to_num(
                    #     np.nanmean(edge_data, axis=0))

        
            coupling_result_dict[f'com{icom}'] = coupling_result
        save_as_pickle(coupling_result_dict,file_path)
    else:
        coupling_result_dict = load_as_pickle(file_path)

    
    slope_metrics = ['slope_pair_comm', 'slope_pair_allcomm']
    strength_metrics = ['strength_pair_comm', 'strength_pair_allcomm']
    LI_coupling = {k: {m: [None]*n_comm for m in slope_metrics+strength_metrics} 
                   for k in ['inte', 'segre']}
    LI_static = {
    k: {
        # Initialize 2D arrays for strength-based metrics (n_comm x n_net)
        **{m: {'t': np.zeros((n_comm, n_pair)), 'p': np.zeros((n_comm, n_pair))} 
           for m in strength_metrics},
        # Initialize 1D arrays for slope-based metrics (n_comm)
        **{m: {'t': np.zeros(n_comm), 'p': np.zeros(n_comm)} 
           for m in slope_metrics}
        }
        for k in ['inte', 'segre']
    }

    
    def get_LI_indices(LL, RR, LR, RL):
        inte = (LL + LR) - (RR + RL)
        segre = (LL - LR) - (RR - RL)
        return inte, segre
    file_path = f'{output_path}/LI_coupling_k{n_comm}.joblib'
    if not os.path.exists(file_path):
        for icom in range(n_comm):
            print(f"\n>>> Compute LI in Community {icom+1}/{n_comm} ===")
            cmap = grad_cmaps[f'grad_{icom}']
            for iMatric, matric_key in enumerate(strength_metrics):
                # data_key = matric_key.split('_')[2]
                data = coupling_result_dict[f'com{icom}'][matric_key]
                
                inte, segre = get_LI_indices(np.nan_to_num(data['LL']), 
                                             np.nan_to_num(data['RR']), 
                                             np.nan_to_num(data['LR']), 
                                             np.nan_to_num(data['RL']))
                for k, val in zip(['inte', 'segre'], [inte, segre]):
                    LI_coupling[k][matric_key][icom] = val
                    t, p = ttest_1samp(val, 0, axis=0)
                    (LI_static[k][matric_key]['t'][icom], 
                     LI_static[k][matric_key]['p'][icom]) = t, p
            
                    # for k in ['inte', 'segre']:
                    # v2_slopes = [compute_stability_batch(run_data.copy())[0] 
                    #              for run_data in LI_coupling[k]['com_net'][icom]]
                    LI_slope=[]
                    for n in range(n_run):
                         LI_slope.append(compute_stability_batch(val[n])[0])
                    LI_coupling[k][slope_metrics[iMatric]][icom] = np.stack(LI_slope)
        save_as_pickle(LI_coupling,f'{output_path}/LI_coupling_k{n_comm}.joblib')
        save_as_pickle(LI_static,f'{output_path}/LI_static_k{n_comm}.joblib')
        save_as_pickle(LI_slope,f'{output_path}/LI_slope_k{n_comm}.joblib')
    else:
        LI_coupling = load_as_pickle(f'{output_path}/LI_coupling_k{n_comm}.joblib')
        LI_static = load_as_pickle(f'{output_path}/LI_static_k{n_comm}.joblib')
        LI_slope = load_as_pickle(f'{output_path}/LI_slope_k{n_comm}.joblib')
    
    # ---- Figure 5A&B
    print("\n---- Start Figure 5A&B ----")
    # for key, title in zip(['inte', 'segre'], ['integration', 'segregation']):
    #     plot_comprehensive_coupling(key, 
    #                                 title, 
    #                                 LI_coupling, 
    #                                 output_path, 
    #                                 custom_colors, 
    #                                 plot_mode='strength',
    #                                 y_limit=(-0.2,0.2))
        
    #     plot_comprehensive_coupling(key, 
    #                                 title, 
    #                                 LI_coupling, 
    #                                 output_path, 
    #                                 custom_colors, 
    #                                 plot_mode='slope',
    #                                 y_limit=(0,0.8))
    
    strength_inte_comm = LI_coupling['inte']['strength_pair_comm']
    strength_segre_comm = LI_coupling['segre']['strength_pair_comm']
    mean_inte = np.nanmean(np.stack(strength_inte_comm),axis=(0,2))
    mean_segre = np.nanmean(np.stack(strength_segre_comm),axis=(0,2))
    t_inte = np.zeros(7)
    t_segre = np.zeros(7)
    p_inte = np.zeros(7)
    p_segre = np.zeros(7)
    for i in range(7):
        t_inte[i],p_inte[i] = ttest_rel(np.nanmean(strength_inte_comm[i],axis=1), mean_inte)
        t_segre[i],p_segre[i] = ttest_rel(np.nanmean(strength_segre_comm[i],axis=1), mean_segre)
        
    fdrcorrection(p_segre)    
    # ---- Figure 5C
    print("\n---- Start Figure 5C ----")
    from statsmodels.stats.multitest import fdrcorrection
    file_path = f'{output_path}/{dataset}_coupling_contribution_k{n_comm}.joblib'
    if not os.path.exists(file_path):
        community_results = []
        for icom in range(n_comm):
            print(f"Processing Community {icom+1}")
            res = calc_coupling_contribution(
                icom,
                LI_coupling, 
                tril_net_index,
                njobs=40           
            )
            community_results.append(res)
        save_as_pickle(community_results, file_path)
    else:
        community_results = load_as_pickle(file_path)
    
    contri_communities_inte = np.zeros((n_comm+1,n_pair,n_run))
    t_com_inte = np.zeros((n_comm+1,n_pair))
    p_com_inte = np.zeros((n_comm+1,n_pair))
    
    contri_communities_segre = np.zeros((n_comm+1,n_pair,n_run))
    t_com_segre = np.zeros((n_comm+1,n_pair))
    p_com_segre = np.zeros((n_comm+1,n_pair))
    
    for icom in range(n_comm): 
        contri_inte = community_results[icom]['contribution_inte']
        contri_inte_new = community_results[icom]['contribution_inte_new']
        for n in range(n_pair):
            t_com_inte[icom,n],p_com_inte[icom,n] = ttest_rel(contri_inte[:,0], 
                                                              contri_inte_new[:,n])
        contri_communities_inte[icom,:,:] = ((contri_inte_new-contri_inte)/contri_inte).T

        contri_segre = community_results[icom]['contribution_segre']
        contri_segre_new = community_results[icom]['contribution_segre_new']
        for n in range(n_pair):
            t_com_segre[icom,n],p_com_segre[icom,n] = ttest_rel(contri_segre[:,0], 
                                                                contri_segre_new[:,n])
        contri_communities_segre[icom,:,:] = ((contri_segre_new-contri_segre)/contri_segre).T
    
    contri_inte_wb = community_results[0]['contribution_inte_wb']
    contri_inte_wb_new = community_results[0]['contribution_inte_wb_new']     
    contri_communities_inte[-1,:,:] = ((contri_inte_wb_new-contri_inte_wb)/contri_inte_wb).T        
    
    contri_segre_wb = community_results[0]['contribution_segre_wb']
    contri_segre_wb_new = community_results[0]['contribution_segre_wb_new'] 
    contri_communities_segre[-1,:,:] = ((contri_segre_wb_new-contri_segre_wb)/contri_segre_wb).T 
                
    p_com_inte[np.isnan(p_com_inte)]=1    
    reject,p_adj = fdrcorrection(p_com_inte.flatten(),0.05)
    contri_inte_sig = contri_communities_inte.copy()
    contri_inte_sig[p_adj.reshape((n_comm+1,n_pair))>0.05,:]=np.nan   
    
    p_com_segre[np.isnan(p_com_segre)]=1    
    reject,p_adj = fdrcorrection(p_com_segre.flatten(),0.05)
    contri_segre_sig = contri_communities_segre.copy()
    contri_segre_sig[p_adj.reshape((n_comm+1,n_pair))>0.05,:]=np.nan   
        
    pair_labels = [(net_yeo7[i], net_yeo7[j]) for i, j in zip(*tril_net_index)]
    xlabels = [f"{pair_labels[i][0]}-{pair_labels[i][1]}" for i in range(28)]
    
    
    plt.figure(figsize=(10,8))
    # cmap.set_bad(color='white')
    ax = sns.heatmap(pd.DataFrame(data=np.nanmean(contri_inte_sig,axis=2),
                                  index=communities_wb,
                                  columns=xlabels),
                     cmap='RdBu_r',square=True,vmin=-0.05,vmax=0.05,cbar=False,)  
    ax.tick_params(axis='both', which='major', length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=14) 
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,fontsize=14)

    plt.tight_layout()
    plt.savefig(f'{output_path}/png//contri_inte_sig.png',dpi=600) 
    plt.savefig(f'{output_path}/eps//contri_inte_sig.eps',dpi=600)
    
    plt.figure(figsize=(10,8))
    # cmap.set_bad(color='white')
    ax = sns.heatmap(pd.DataFrame(data=np.nanmean(contri_segre_sig,axis=2),
                                  index=communities_wb,
                                  columns=xlabels),
                     cmap='RdBu_r',square=True,vmin=-0.05,vmax=0.05,cbar=False)  
    ax.tick_params(axis='both', which='major', length=0)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90,fontsize=14) 
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,fontsize=14)
    plt.tight_layout()
    plt.savefig(f'{output_path}/png//contri_segre_sig.png',dpi=600) 
    plt.savefig(f'{output_path}/eps//contri_segre_sig.eps',dpi=600)
    
    # ---- Figure 5D
    print("\n---- Start Figure 5D ----")

    df = pd.DataFrame({'slope of integration':np.nanmean(contri_inte_sig,axis=2).flatten(),
                       'slope of segregation':np.nanmean(contri_segre_sig,axis=2).flatten()})
    plot_joint_correlation(df, 
                           'slope of integration', 
                           'slope of segregation', 
                           [-0.06,0.06],
                           [-0.06,0.06],
                           output_path,
                           
                           'corr_contri_inte_vs_segre_sig')
    
    
    # ---- Figure 5E & F
    print("\n---- Start Figure 5E & F ----")
    contri_matrix = np.nanmean(contri_inte_sig,axis=2)
    rho_inte,p_rho = spearmanr(contri_matrix.T[:,:-1],nan_policy='omit')
    df = pd.DataFrame(data=rho_inte,index=communities,columns=communities)
    g=sns.clustermap(df,cmap='RdBu_r',figsize=(4,4),cbar=False,cbar_kws={'label': ''},vmin=-1,vmax=1)
    # g.ax_row_dendrogram.set_visible(False)
    # g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=14, rotation=0)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=14, rotation=0)
    g.ax_heatmap.tick_params(axis='both', which='major', length=0)
    g.ax_cbar.set_ylabel('')
    g.ax_cbar.set_xticks([])
    g.ax_cbar.set_yticks([])
    g.fig.savefig(f'{output_path}/png/clustermap_contri_inte_sig.png',dpi=600)
    g.fig.savefig(f'{output_path}/eps/clustermap_contri_inte_sig.eps',dpi=600)
    plt.show()
    
    contri_matrix = np.nanmean(contri_segre_sig,axis=2)
    rho_segre,p_rho = spearmanr(contri_matrix.T[:,:-1],nan_policy='omit')
    df = pd.DataFrame(data=rho_segre,index=communities,columns=communities)
    
    g=sns.clustermap(df,cmap='RdBu_r',figsize=(4,4),cbar=False,cbar_kws={'label': ''},vmin=-1,vmax=1)
    # g.ax_row_dendrogram.set_visible(False)
    # g.ax_col_dendrogram.set_visible(False)
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), fontsize=14, rotation=0)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=14, rotation=0)
    g.ax_heatmap.tick_params(axis='both', which='major', length=0)
    g.ax_cbar.set_ylabel('')
    g.ax_cbar.set_xticks([])
    g.ax_cbar.set_yticks([])
    g.fig.savefig(f'{output_path}/png/clustermap_contri_segre_sig.png',dpi=600)
    g.fig.savefig(f'{output_path}/eps/clustermap_contri_segre_sig.eps',dpi=600)
    plt.show()
    
    LI4_obs = load_as_pickle(f'{output_path}/LI4_obs_k{n_comm}.joblib')
    community_inte = np.stack([LI4_obs['segre'][i][tril_net_index] for i in range(n_comm)])
    rho1,p_rho1 = spearmanr(community_inte.T,nan_policy='omit')
    
    df = pd.DataFrame({'contri similarity':rho_inte.flatten(),
                       'Community lateralization pattern similarity':rho1.flatten()})
    
    plot_joint_correlation(df, 
                           'contri similarity', 
                           'Community lateralization pattern similarity', 
                           [-1,1],
                           [-1,1],
                           output_path,
                           'corr_contri_vs_community_inte_sig')
    
    community_segre = np.stack([LI4_obs['segre'][i][tril_net_index] for i in range(n_comm)])
    rho1,p_rho1 = spearmanr(community_segre.T,nan_policy='omit')
    df = pd.DataFrame({'contri similarity':rho_segre.flatten(),
                       'Community lateralization pattern similarity':rho1.flatten()})
    
    plot_joint_correlation(df, 
                           'contri similarity', 
                           'Community lateralization pattern similarity', 
                           [-1,1],
                           [-1,1],
                           output_path,
                           'corr_contri_vs_community_segre_sig')
    
    
    
def run_community_edge_part_pipeline(
    ci_matrix_dicts,
    ci_matrix_null_dicts,
    labels_dict, 
    output_path,
    n_perm=10000, 
):

    # 1. Basic Parameters & Metadata ---
    n_net = 7
    # Calculate number of communities (subtracting 0/noise if exists)
    n_comm = len(np.unique(ci_matrix_dicts['cortex'])) - 1
    communities_sample = [f'C{i}' for i in range(1, n_comm + 1)]
    
    # print(f'n_comm = {n_comm}')
    # Define network label sets
    net_yeo7 = ['DMN', 'FPN', 'LIMB', 'VAN', 'DAN', 'SMN', 'VIS']
    net_yeo7x2 = [f'L_{n}' for n in net_yeo7] + [f'R_{n}' for n in net_yeo7]
    hetero_nets = ['DMN', 'FPN', 'LIMB', 'VAN', 'DAN']
    net_to_type = {n: ('H' if n in hetero_nets else 'U') for n in net_yeo7}

    # Pre-calculate edge distribution counts
    community_edge_counts = calc_community_network_pair_counts(
        ci_matrix_dicts['cortex'],
        np.concatenate((labels_dict['yeo7']['L_cortex'], labels_dict['yeo7']['R_cortex'])), 
        n_comm
    )

    # 2. Observe & Null Matrix Initialization ---
    ci_network_dict = {'mean': {ftype: np.zeros((n_comm, n_net, n_net)) for ftype in ['LL', 'RR', 'LR', 'RL']}}
    ci_network_dict['mean']['cortex'] = np.zeros((n_comm, n_net * 2, n_net * 2))
    
    ci_network_null_dict = {'mean': {ftype: np.zeros((n_comm, n_net, n_net, n_perm)) 
                                    for ftype in ['LL', 'RR', 'LR', 'RL']}}

    for icom in range(n_comm):
         
        # Isolate edges for the current community
        ci_matrix_com = (ci_matrix_dicts['cortex'] == (icom + 1)).astype(int)

        # Aggregate observed edges to network matrices
        out = aggregate_edge_to_network(
            ci_matrix_com,
            row_labels=labels_dict['yeo7']['bil_cortex'],
            col_labels=labels_dict['yeo7']['bil_cortex'],
            network_names=net_yeo7x2,
            metrics=("mean", "std", 'num'),
            ftype='cortex'
        )
        ci_network_mean = out[0]['fc_network_mean']
        ci_network_dict['mean']['LL'][icom, :, :] = ci_network_mean[:n_net, :n_net].T
        ci_network_dict['mean']['RR'][icom, :, :] = ci_network_mean[n_net:, n_net:].T
        ci_network_dict['mean']['LR'][icom, :, :] = np.tril(ci_network_mean[:n_net, n_net:])
        ci_network_dict['mean']['RL'][icom, :, :] = np.triu(ci_network_mean[:n_net, n_net:]).T
        ci_network_dict['mean']['cortex'][icom, :, :] = ci_network_mean.T

    # 3. Community-to-Network Mapping Loop ---
    file_path = f'{output_path}/REST_ci_network_null_dict_k{n_comm}.joblib'
    ci_network_null_dict = load_as_pickle(f'{output_path}/REST_ci_network_null_dict_k{n_comm}.joblib')
    
    # 4. Lateralization Index (LI) Analysis ---
    types = ['inte', 'segre']
    LI4_obs = {t: np.zeros((n_comm, n_net, n_net)) for t in types}
    for t in types: LI4_obs[f'{t}_whole'] = np.zeros(n_comm)
    
    LI4_obs_sig = {t: np.zeros((n_comm, n_net, n_net)) for t in types}
    for t in types: LI4_obs_sig[f'{t}_whole'] = np.zeros(n_comm)

    LI4_null = {t: {'mean_pair': {f'community{i+1}': np.zeros((n_perm, n_net, n_net)) 
                                  for i in range(n_comm)}} for t in types}
    LI4_pvalue = {t: {'mean_pair': []} for t in types}

    key = 'mean'
    for icom in range(n_comm):
        # Observed quadrants
        RR, LL = ci_network_dict[key]['RR'][icom], ci_network_dict[key]['LL'][icom]
        LR, RL = ci_network_dict[key]['LR'][icom], ci_network_dict[key]['RL'][icom]
        
        # Formulas for LI
        LI4_obs['inte'][icom] = (LL + LR - RR - RL)
        LI4_obs['segre'][icom] = (LL - LR - RR + RL)


        # Aggregate Null distributions
        for n in range(n_perm):
            n_RR, n_LL = ci_network_null_dict[key]['RR'][icom,:,:,n], ci_network_null_dict[key]['LL'][icom,:,:,n]
            n_LR, n_RL = ci_network_null_dict[key]['LR'][icom,:,:,n], ci_network_null_dict[key]['RL'][icom,:,:,n]
            LI4_null['inte']['mean_pair'][f'community{icom+1}'][n] = (n_LL + n_LR - n_RR - n_RL)
            LI4_null['segre']['mean_pair'][f'community{icom+1}'][n] = (n_LL - n_LR - n_RR + n_RL)

        # Significance testing
        for t in types:
            p_vals = calc_two_tailed_p(LI4_obs[t][icom], LI4_null[t]['mean_pair'][f'community{icom+1}'], True)
            LI4_pvalue[t]['mean_pair'].append(p_vals)
            
            sig_data = LI4_obs[t][icom].copy()
            sig_data[p_vals > 0.05] = np.nan
            LI4_obs_sig[t][icom] = sig_data
            
            # Weighted averages
            LI4_obs[f'{t}_whole'][icom] = calc_weighted_avg(LI4_obs[t][icom], community_edge_counts[icom])
            LI4_obs_sig[f'{t}_whole'][icom] = calc_weighted_avg(sig_data, community_edge_counts[icom])

    return LI4_obs, LI4_obs_sig
    
def run_community_node_part_pipeline(ci_matrix, 
                                     ci_matrix_null, 
                                     labels_dict,
                                     output_path,
                                     n_perm=10000,
):
    
   
    n_net = 7
    n_comm = len(np.unique(ci_matrix))-1
    net_yeo7 = ['DMN', 'FPN', 'LIMB', 'VAN', 'DAN','SMN', 'VIS']
    community_node_results = analyze_node_profiles(
        ci_matrix, 
        labels_dict, 
        net_yeo7,
        './', 
        prefix="TEMP",
        custom_cmap='PuBu',
        plot_figure=False
    )

    file_path = f'{output_path}/REST_community_node_results_null_k{n_comm}.joblib'
    community_node_results_null = load_as_pickle(file_path)
    
    community_node_affi_all = {
        ftype: community_node_results[ftype]['affiliation']
        for ftype in ['LL', 'RR', 'LR', 'RL']
    }
    community_node_affi_all_null = {
        ftype: np.stack([community_node_results_null[i][ftype]['affiliation']
                         for i in range(n_perm)])
        for ftype in ['LL', 'RR', 'LR', 'RL']
    }

    network_result = analyze_network_alliliation_profiles(
        community_node_affi_all,
        community_node_affi_all_null,
        labels_dict,
        net_yeo7,
        '',
        output_path,
        dataset='REST'
    )
 
   
    def apply_fdr_correction(p_df, alpha=0.05):
        """
        ¶Ô DataFrame ¸ñÊ½µÄ P Öµ¾ØÕó½øÐÐ FDR ÐÞÕý¡£
        
        ²ÎÊý:
        - p_df: pd.DataFrame, Ô­Ê¼ P Öµ¾ØÕó
        - alpha: float, ÏÔÖøÐÔË®Æ½ (Ä¬ÈÏ 0.05)
        
        ·µ»Ø:
        - sig_mask: pd.DataFrame (bool), ÏÔÖøÐÔÑÚÂë¾ØÕó (True ±íÊ¾ÏÔÖø)
        - p_adj_df: pd.DataFrame (float), ÐÞÕýºóµÄ P Öµ¾ØÕó
        """
        n_rows, n_cols = p_df.shape
        index_names = p_df.index
        column_names = p_df.columns
    
        reject, p_adj = fdrcorrection(p_df.values.flatten(), alpha=alpha)
        
        sig_mask = pd.DataFrame(reject.reshape((n_rows, n_cols)), 
                                 index=index_names, columns=column_names)
        
        p_adj_df = pd.DataFrame(p_adj.reshape((n_rows, n_cols)), 
                                 index=index_names, columns=column_names)
        
        return sig_mask, p_adj_df
    
    (network_result['sig_inte'], 
     network_result['padj_inte']) = apply_fdr_correction(network_result['p_inte'], 
                                                         alpha=0.05)

    (network_result['sig_segre'], 
     network_result['padj_segre']) = apply_fdr_correction(network_result['p_segre'], 
                                                         alpha=0.05)

    return  network_result
    

def run_community_coupling_part_pipeline(
    ci_matrix_dicts,
    arousal_coupling_matrix,
    labels_dict,
    ):
    
    
    # --- 0. Parameters & Setup ---
    net_yeo7 = ['DMN', 'FPN', 'LIMB', 'VAN', 'DAN', 'SMN', 'VIS']
    net_yeo7x2 = [f'L_{n}' for n in net_yeo7] + [f'R_{n}' for n in net_yeo7]
    n_comm = len(np.unique(ci_matrix_dicts['cortex'])) - 1
    n_run, n_roi = arousal_coupling_matrix.shape[0], 200
    n_net, n_pair = 7, 28
    tril_net_index = np.tril_indices(n_net)
    triu_edge_index = np.triu_indices(n_roi)
    ftype_list = ['LL', 'RR', 'LR', 'RL']
      
    coupling_result_dict = {f'C{icom}': {} for icom in range(n_comm)}  
    communities = [f'C{i}' for i in range(1,n_comm+1)]  
    communities_wb = np.hstack((communities,'wb'))
    
    coupling_data = {m: {f: np.full((n_comm, n_net, n_net), np.nan) for f in ftype_list} 
                         for m in ['comm', 'allcomm']}
    coupling_data['edge'] = {f: np.full((n_comm, n_roi, n_roi), np.nan) for f in ftype_list}
    
    def compute_stability_batch(data_vect):
        data_vect[data_vect == 0] = np.nan
        notnan_idx = ~np.isnan(data_vect)
        if not np.any(notnan_idx): return 0, 0
        vals_sorted = np.sort(data_vect[notnan_idx])[::-1]
        return compute_stability(vals_sorted)
    

    for icom in range(n_comm):

        
        mask_com = (ci_matrix_dicts['cortex'] == (icom + 1)).astype(float)
        coupling_result = {
            # 'slope_net': {ftype: np.zeros(n_run) for ftype in ftype_list},
            # 'slope_whole':{ftype: np.zeros(n_run) for ftype in ftype_list},
            'strength_pair_allcomm': {ftype: np.zeros((n_run, n_pair)) for ftype in ftype_list},
            'strength_pair_comm': {ftype: np.zeros((n_run, n_pair)) for ftype in ftype_list},
            # 'strength_allnet': {ftype: np.zeros((n_run, n_pair)) for ftype in ftype_list},
            # 'strength_node': {ftype: np.zeros((n_run, n_roi)) for ftype in ftype_list}
        }
        for i in range(n_run):
            
            curr_coupling = arousal_coupling_matrix[i].copy()
            curr_coupling_com = curr_coupling * mask_com
            # curr_coupling_com = curr_coupling_com + curr_coupling_com.T
            curr_coupling_com[curr_coupling_com == 0] = np.nan
            
            out_comm = aggregate_edge_to_network(
                curr_coupling_com, 
                labels_dict['yeo7']['bil_cortex'],          
                labels_dict['yeo7']['bil_cortex'], 
                metrics=("mean", "std", 'num'),
                network_names=net_yeo7x2, 
                ftype='cortex')
            
            net_mean = out_comm[0]['fc_network_mean']
            coupling_data['comm']['LL'][icom] = net_mean[:n_net,:n_net].T
            coupling_data['comm']['RR'][icom] = net_mean[n_net:,n_net:].T
            coupling_data['comm']['LR'][icom] = np.tril(net_mean[:n_net,n_net:])
            coupling_data['comm']['RL'][icom] = np.triu(net_mean[:n_net,n_net:]).T
            coupling_data['comm']['LR'][icom][coupling_data['comm']['LR'][icom]==0]=np.nan
            coupling_data['comm']['RL'][icom][coupling_data['comm']['RL'][icom]==0]=np.nan
                             
            out_all = aggregate_edge_to_network(
                curr_coupling,
                row_labels=labels_dict['yeo7']['bil_cortex'],
                col_labels=labels_dict['yeo7']['bil_cortex'],
                network_names=net_yeo7x2,
                metrics=("mean", "std", 'num'),
                ftype='cortex'
            )
            allcomm_mean = out_all[0]['fc_network_mean']
            coupling_data['allcomm']['LL'][icom] = allcomm_mean[:n_net,:n_net].T
            coupling_data['allcomm']['RR'][icom] = allcomm_mean[n_net:,n_net:].T
            coupling_data['allcomm']['LR'][icom] = np.tril(allcomm_mean[:n_net,n_net:])
            coupling_data['allcomm']['RL'][icom] = np.triu(allcomm_mean[:n_net,n_net:]).T
            coupling_data['allcomm']['LR'][icom][coupling_data['allcomm']['LR'][icom]==0]=np.nan
            coupling_data['allcomm']['RL'][icom][coupling_data['allcomm']['RL'][icom]==0]=np.nan
            
                
            for f in ftype_list:
                for key in ['comm', 'allcomm']:
                    # Extract vector using indices
                    vec = coupling_data[key][f][icom][tril_net_index]
                    coupling_result[f'strength_pair_{key}'][f][i] = vec
            
    
        coupling_result_dict[f'com{icom}'] = coupling_result

    slope_metrics = ['slope_pair_comm', 'slope_pair_allcomm']
    strength_metrics = ['strength_pair_comm', 'strength_pair_allcomm']
    LI_coupling = {k: {m: [None]*n_comm for m in slope_metrics+strength_metrics} 
                   for k in ['inte', 'segre']}
    LI_static = {
    k: {
        # Initialize 2D arrays for strength-based metrics (n_comm x n_net)
        **{m: {'t': np.zeros((n_comm, n_pair)), 'p': np.zeros((n_comm, n_pair))} 
           for m in strength_metrics},
        # Initialize 1D arrays for slope-based metrics (n_comm)
        **{m: {'t': np.zeros(n_comm), 'p': np.zeros(n_comm)} 
           for m in slope_metrics}
        }
        for k in ['inte', 'segre']
    }

    
    def get_LI_indices(LL, RR, LR, RL):
        inte = (LL + LR) - (RR + RL)
        segre = (LL - LR) - (RR - RL)
        return inte, segre

    for icom in range(n_comm):

        for iMatric, matric_key in enumerate(strength_metrics):
            # data_key = matric_key.split('_')[2]
            data = coupling_result_dict[f'com{icom}'][matric_key]
            
            inte, segre = get_LI_indices(np.nan_to_num(data['LL']), 
                                         np.nan_to_num(data['RR']), 
                                         np.nan_to_num(data['LR']), 
                                         np.nan_to_num(data['RL']))
            for k, val in zip(['inte', 'segre'], [inte, segre]):
                LI_coupling[k][matric_key][icom] = val
                t, p = ttest_1samp(val, 0, axis=0)
                (LI_static[k][matric_key]['t'][icom], 
                 LI_static[k][matric_key]['p'][icom]) = t, p
        
                # for k in ['inte', 'segre']:
                # v2_slopes = [compute_stability_batch(run_data.copy())[0] 
                #              for run_data in LI_coupling[k]['com_net'][icom]]
                LI_slope=[]
                for n in range(n_run):
                     LI_slope.append(compute_stability_batch(val[n])[0])
                LI_coupling[k][slope_metrics[iMatric]][icom] = np.stack(LI_slope)
    
    return LI_coupling
