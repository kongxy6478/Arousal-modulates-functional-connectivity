import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from analysis.network_mapping import aggregate_node_to_network
import sys
from scipy.stats import spearmanr
sys.path.insert(0, "/data/disk0/kongxiangyu/result_explanation/GS_new/FCw_coupling_project")
sys.path.append('/data/disk0/kongxiangyu/result_explanation/trans/')
sys.path.extend([
    '/data/disk0/kongxiangyu/result_explanation/GS_new/',
    '/data/disk0/kongxiangyu/result_explanation/code/util/'
])
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

def get_k_config(k):

    configs = {
        5: ['#2364AA', '#3DA5D9', '#FEC601', '#EA7317', '#F94144'],
        6: ['#2364AA', '#3DA5D9', '#73BFB8', '#FEC601', '#EA7317', '#D64550'],
        7: ['#2364AA', '#3DA5D9', '#73BFB8', '#B9C35D', '#FEC601', '#EA7317', '#D64550']
    }

    if k not in configs:
        raise ValueError(f"Unsupported k={k}. Please use 5, 6, or 7.")

    colors = configs[k]
    
    cmap = ListedColormap(colors, name=f'my_cmap_k{k}')
    
    grad_cmaps = {
        f"grad_{i}": LinearSegmentedColormap.from_list(f"g{i}", ['#ffffff', color])
        for i, color in enumerate(colors)
    }

    return colors, cmap, grad_cmaps

def calc_node_entropy(mat):
    """
    Computes node-level entropy based on a square matrix.
    mat: (N, N) array
    """
    N = mat.shape[0]
    max_ci = int(np.nanmax(mat))
    h = np.zeros((N, max_ci))
    
    for i in range(N):
        # For both Intra and Inter, we simply look at the connections of node i
        # If it's Inter-hemispheric (LR), mat[i, :] contains all links from L-node i to R-nodes
        node_connections = mat[i, :]
        valid_mask = ~np.isnan(node_connections) & (node_connections > 0)
        
        if np.any(valid_mask):
            counts, _ = np.histogram(node_connections[valid_mask], 
                                     bins=np.arange(1, max_ci + 2))
            h[i, :] = counts
            
    # Calculate entropy
    with np.errstate(divide='ignore', invalid='ignore'):
        p = h / np.sum(h, axis=1, keepdims=True)
        entropy = -np.nansum(p * np.log(p + 1e-12), axis=1)
        # Normalize by log(K) where K is number of possible communities
        enorm = entropy / np.log(max_ci)
        
    return h, enorm

def calc_node_affiliation(mat, threshold=0.2):
    """
    Calculate the proportion of edges connected to each node that belong to 
    each edge community.
    
    Parameters:
    -----------
    mat : ndarray
        Square matrix (N x N) where entries are edge community labels.
    threshold : float
        Proportion threshold (0 to 1). Values below this are set to NaN.
        
    Returns:
    --------
    A_thr : ndarray
        Thresholded affiliation matrix (n_comm x N).
    A : ndarray
        Original affiliation matrix (n_comm x N).
    """
    # Ensure matrix is integer and clean NaNs
    mat = np.nan_to_num(mat).astype(int)
    N = mat.shape[0]
    n_comm = int(np.max(mat))
    
    # A[k, i] will store the proportion of edges of node i that belong to community k+1
    A = np.zeros((n_comm, N))
    
    for i in range(N):
        # Get community labels for all edges connected to node i
        # Exclude self-loop by ignoring diagonal
        row_values = mat[i, :]
        valid_edges = row_values[row_values > 0]
        
        if valid_edges.size > 0:
            # Use bincount for faster frequency counting
            counts = np.bincount(valid_edges, minlength=n_comm + 1)
            # Normalization: counts[1:] corresponds to communities 1 to n_comm
            A[:, i] = counts[1:] / len(valid_edges)
            
    # Apply threshold
    A_thr = np.where(A > threshold, A, np.nan)
    
    return A_thr, A

def calc_edge_communities_to_nodes(ci, lab, net, prefix=None, hemi_prefix=None, 
                                   output_path=None, cmap='tab20', threshold=0.2, 
                                   plot_figure=False, **kwargs):
    """
    Refactored analysis function that separates computation and plotting.
    """
    N = len(lab)
    
    # 1. Map edge communities into full matrix
    mat = np.zeros((N, N))
    
    expected_tri_len = N * (N - 1) // 2
    if len(ci) == expected_tri_len:
        tri_mask = np.triu(np.ones((N, N)), 1).astype(bool)
        mat[tri_mask] = ci
        mat = mat + mat.T  
    elif len(ci) == N * N:
        mat = ci.reshape(N, N)
        
    # tri_mask = np.triu(np.ones((N, N)), 1).astype(bool)
    # mat[tri_mask] = ci
    # mat = mat + mat.T
    
    # 2. Compute affiliation (Calculation Part)
    A_thr, A = calc_node_affiliation(mat, threshold=threshold)
    
    # 3. Compute normalized node entropy (Assuming external function exists)
    u, v = np.triu_indices(N, 1)
    _, enorm = calc_node_entropy(mat)

    # 4. Visualization (Plotting Part)
    if plot_figure:
        plot_node_affiliation(
            A_thr=A_thr,
            prefix=prefix,
            hemi_prefix=hemi_prefix,
            output_path=output_path,
            threshold=threshold,
            cmap_name=cmap
        )

    return {
        'affiliation_thr': A_thr,
        'node_entropy': enorm,
        'affiliation': A
    }

import pandas as pd
def calc_modality_profiles(arr, network_names, net_to_type):
    """
    Extract every individual edge value categorized by modality types (U-U, H-H, U-H).
    Returns a long-format DataFrame where each row is a single connection (edge).
    """
    n_net = len(network_names)
    
    # --- 1. Create Logical Masks for Modality Groups ---
    u_mask = np.array([net_to_type[n] in ['U', 'uni'] for n in network_names])
    h_mask = np.array([net_to_type[n] in ['H', 'hetero'] for n in network_names])
    
    mask_uu = np.outer(u_mask, u_mask)
    mask_hh = np.outer(h_mask, h_mask)
    mask_uh = np.outer(u_mask, h_mask) | np.outer(h_mask, u_mask)
    
    # Use lower triangle mask to get unique edges (no self-connections if k=-1)
    tri_mask = np.tril(np.ones((n_net, n_net), dtype=bool), k=-1)
    
    # Final indices for each category
    type_masks = {
        'U - U': mask_uu & tri_mask,
        'H - H': mask_hh & tri_mask,
        'U - H': mask_uh & tri_mask
    }

    all_edges_list = []

    # --- 2. Extract Raw Values ---
    if arr.ndim == 3:
        # Observed Data: (n_com, n_net, n_net)
        n_com = arr.shape[0]
        for icom in range(n_com):
            mat = arr[icom]
            for label, msk in type_masks.items():
                # Extract all values under the mask
                edge_values = mat[msk]
                
                # Filter out NaNs or zeros if they represent "no connection"
                # (Keep 0 if it's a valid coupling value, otherwise filter it)
                valid_edges = edge_values[~np.isnan(edge_values)]
                
                if len(valid_edges) > 0:
                    temp_df = pd.DataFrame({
                        'community': f'C{icom+1}',
                        'modal': label,
                        'value': valid_edges
                    })
                    all_edges_list.append(temp_df)

    elif arr.ndim == 4:
        # Null Data: (n_com, n_net, n_net, n_perm)
        n_com, _, _, n_perm = arr.shape
        for icom in range(n_com):
            for iperm in range(n_perm):
                mat = arr[icom, :, :, iperm]
                for label, msk in type_masks.items():
                    edge_values = mat[msk]
                    valid_edges = edge_values[~np.isnan(edge_values)]
                    
                    if len(valid_edges) > 0:
                        all_edges_list.append(pd.DataFrame({
                            'community': f'C{icom+1}',
                            'perm_idx': iperm,
                            'modal': label,
                            'value': valid_edges
                        }))
    
    # Combine all individual edge DataFrames
    df_final = pd.concat(all_edges_list, ignore_index=True)
    return df_final

def calc_two_tailed_p(obs, null_dist, fdr_flag=False):

    obs = np.asarray(obs)
    null_dist = np.asarray(null_dist)

    n_perm = null_dist.shape[0]

    # Raw permutation p
    extreme = np.sum(np.abs(null_dist) >= np.abs(obs), axis=0)
    p_raw = (extreme + 1) / (n_perm + 1)

    if not fdr_flag:
        return p_raw

    # ---------- FDR correction (BH) ----------
    # Flatten
    p_flat = p_raw.reshape(-1)
    n = len(p_flat)

    # argsort
    order = np.argsort(p_flat)
    ranked_p = p_flat[order]

    # BH step-up
    bh = ranked_p * n / (np.arange(1, n+1))

    # enforce monotonicity
    bh = np.minimum.accumulate(bh[::-1])[::-1]

    # put back original shape
    p_fdr = np.empty_like(p_flat)
    p_fdr[order] = np.minimum(bh, 1.0)

    return p_fdr.reshape(p_raw.shape)


def calc_LI_entropy(LI4_obs_sig, 
                    community_edge_counts, 
                    LI_type='inte'):
    """
    Calculate the Shannon entropy of contribution weights for each community.
    
    Scientific Significance:
    -----------------------
    Measures the functional diversity/specialization of each edge community 
    based on the distribution of significant network-pair contributions.

    Parameters:
    -----------
    LI4_obs_sig : dict
        Significant LI values per community and network-pair (Non-significant are NaN).
    community_edge_counts : ndarray
        Edge counts per community and network-pair.
    LI_type : str
        Type of index to use ('inte' or 'segre').

    Returns:
    --------
    entropy_array : ndarray
        Entropy value for each community.
    """
    n_comm = LI4_obs_sig[LI_type].shape[0]
    entropy_array = np.zeros(n_comm)

    for c in range(n_comm):
        # 1. Identify significant network-pairs using the mask
        sig_mask = ~np.isnan(LI4_obs_sig[LI_type][c])
        
        if np.sum(sig_mask) == 0:
            entropy_array[c] = np.nan
            continue

        # 2. Calculate contribution weight = edge counts * LI value
        # Retaining original variable name: contrib
        contrib = LI4_obs_sig[LI_type][c] * community_edge_counts[c] * sig_mask

        # 3. Normalize to a probability distribution p
        # Flatten and filter to avoid log2(0) or negative values
        p = contrib.flatten()
        p = p[p > 0] 
        
        if p.size == 0:
            entropy_array[c] = np.nan
            continue
            
        p = p / np.sum(p)

        # 4. Compute Shannon Entropy: H = -sum(p * log2(p))
        entropy_array[c] = -np.sum(p * np.log2(p))

    return entropy_array

def calc_coupling_contribution(icom, 
                               LI_coupling, 
                               tril_net_index, 
                               njobs=10):
    """
    Perform Leave-One-Out (LOO) sensitivity analysis to compute the contribution 
    of each network pair to the community and whole-brain coupling slope.
    
    Scientific Significance:
    -----------------------
    By dropping one network pair at a time and recalculating the slope, we identify 
    which specific regional interactions are the 'drivers' of community stability.
    """
    # Import locally to ensure availability within parallel workers
    from analysis.slope import compute_stability

    # ----  Initialization ----
    n_pairs = len(tril_net_index[0]) # Should be 28 for Yeo7
    n_subjects = 485  # Standard sample size for this dataset
    
    # Pre-allocate result containers (Subject x Pair)
    # Using original variable names as requested
    contribution_inte = np.zeros((n_subjects, n_pairs))
    contribution_segre = np.zeros((n_subjects, n_pairs))
    contribution_inte_wb = np.zeros((n_subjects, n_pairs))
    contribution_segre_wb = np.zeros((n_subjects, n_pairs))
    contribution_inte_new = np.zeros((n_subjects, n_pairs))
    contribution_segre_new = np.zeros((n_subjects, n_pairs))
    contribution_inte_wb_new = np.zeros((n_subjects, n_pairs))
    contribution_segre_wb_new = np.zeros((n_subjects, n_pairs))

    def _process_subject_loo(i):
        """Internal helper to process a single subject's LOO analysis."""
        
        # Local results for one subject
        c_segre = np.zeros(n_pairs)
        c_inte = np.zeros(n_pairs)
        c_segre_wb = np.zeros(n_pairs)
        c_inte_wb = np.zeros(n_pairs)
        c_segre_new = np.zeros(n_pairs)
        c_inte_new = np.zeros(n_pairs)
        c_segre_wb_new = np.zeros(n_pairs)
        c_inte_wb_new = np.zeros(n_pairs)

        # Helper: Calculate slope from vector with optional drop index
        def _get_slope(data_vect, drop_idx=None):
            v = data_vect.copy()
            if drop_idx is not None:
                v[drop_idx] = np.nan
            v[v == 0] = np.nan
            valid_vals = v[~np.isnan(v)]
            if len(valid_vals) < 2: return np.nan
            # Sort descending for stability/slope calculation
            sorted_vals = np.sort(valid_vals)[::-1]
            slope, _ = compute_stability(sorted_vals)
            return slope

        # 1. Baseline: Compute original slopes (Full set)
        orig_inte = _get_slope(LI_coupling['inte']['strength_pair_comm'][icom][i])
        orig_segre = _get_slope(LI_coupling['segre']['strength_pair_comm'][icom][i])
        orig_inte_wb = _get_slope(LI_coupling['inte']['strength_pair_allcomm'][0][i])
        orig_segre_wb = _get_slope(LI_coupling['segre']['strength_pair_allcomm'][0][i])

        # 2. Iteration: Drop each pair one by one
        for k in range(n_pairs):
            c_inte[k] = orig_inte
            c_segre[k] = orig_segre
            c_inte_wb[k] = orig_inte_wb
            c_segre_wb[k] = orig_segre_wb
            
            # Recalculate with index 'k' removed
            c_inte_new[k] = _get_slope(LI_coupling['inte']['strength_pair_comm'][icom][i], k)
            c_segre_new[k] = _get_slope(LI_coupling['segre']['strength_pair_comm'][icom][i], k)
            c_inte_wb_new[k] = _get_slope(LI_coupling['inte']['strength_pair_allcomm'][0][i], k)
            c_segre_wb_new[k] = _get_slope(LI_coupling['segre']['strength_pair_allcomm'][0][i], k)

        return (c_segre, c_inte, c_segre_wb, c_inte_wb, 
                c_segre_new, c_inte_new, c_segre_wb_new, c_inte_wb_new)

    # ---- Parallel Execution ----
    with Parallel(n_jobs=njobs, backend='loky') as parallel:
        results = parallel(
            delayed(_process_subject_loo)(i) 
            for i in tqdm(range(n_subjects), desc=f"LOO Analysis - Community {icom+1}")
        )

    # ---- Result Aggregation ----
    for i, res in enumerate(results):
        contribution_segre[i, :] = res[0]
        contribution_inte[i, :] = res[1]
        contribution_segre_wb[i, :] = res[2]
        contribution_inte_wb[i, :] = res[3]
        contribution_segre_new[i, :] = res[4]
        contribution_inte_new[i, :] = res[5]
        contribution_segre_wb_new[i, :] = res[6]
        contribution_inte_wb_new[i, :] = res[7]

    # Return using exact original dictionary keys
    return {
        'contribution_segre': contribution_segre,
        'contribution_inte': contribution_inte,
        'contribution_segre_wb': contribution_segre_wb,
        'contribution_inte_wb': contribution_inte_wb,
        'contribution_segre_new': contribution_segre_new,
        'contribution_inte_new': contribution_inte_new,
        'contribution_segre_wb_new': contribution_segre_wb_new,
        'contribution_inte_wb_new': contribution_inte_wb_new
    }


def calc_community_network_pair_counts(edge_label, roi_network, n_communities=7):
    """
    Map edge community assignments to canonical brain network pairs and count their distribution.
    
    Scientific Significance:
    -----------------------
    Quantifies the 'anatomical footprint' of each edge community by counting how many 
    edges within a community fall into specific network-to-network interactions 
    (e.g., DMN-FPN, VIS-VIS).
    """
    # 1. Standardize network labels
    unique_nets = ['DMN', 'FPN', 'LIMB', 'VAN', 'DAN', 'SMN', 'VIS']
    n_networks = len(unique_nets)

    # Map string labels to integer indices
    net_to_int = {lab: i for i, lab in enumerate(unique_nets)}
    roi_network_int = np.array([net_to_int[x] for x in roi_network])

    # 2. Initialize counts (n_comm, n_net, n_net)
    counts = np.zeros((n_communities, n_networks, n_networks), dtype=int)

    # 3. Extract edge information for the lower triangle
    n = edge_label.shape[0]
    roi_i, roi_j = np.tril_indices(n, k=-1)

    # Get community IDs and network indices for each edge
    comms = edge_label[roi_i, roi_j].astype(int)
    ni = roi_network_int[roi_i]
    nj = roi_network_int[roi_j]

    # 4. Statistical aggregation
    # We use a vectorized approach where possible, or a more efficient loop
    for c_val in range(1, n_communities + 1):
        c_idx = c_val - 1
        # Filter edges belonging to the current community
        mask = (comms == c_val)
        if not np.any(mask):
            continue
            
        c_ni = ni[mask]
        c_nj = nj[mask]

        # Count occurrences for each network pair within this community
        for i, j in zip(c_ni, c_nj):
            # Ensure consistent upper-triangle indexing
            idx_i, idx_j = (i, j) if i < j else (j, i)
            counts[c_idx, idx_i, idx_j] += 1
        
    # 5. Mirror the matrix to get a symmetric/transposed representation as per original logic
    for c in range(n_communities):
        counts[c] = counts[c].T

    return counts

def calc_weighted_avg(data, counts):
    mask = ~np.isnan(data)
    w_sum = np.nansum(data * counts * mask)
    t_edges = np.nansum(counts * mask)
    return w_sum / t_edges if t_edges > 0 else 0

def plot_node_affiliation(A_thr, 
                          prefix, 
                          hemi_prefix, 
                          output_path, 
                          threshold=0.2, 
                          cmap_name='tab20'):
    """
    Visualize node-level affiliation as a color-coded matrix.
    
    Parameters:
    -----------
    A_thr : ndarray
        The thresholded affiliation matrix from calculate_community_affiliation.
    prefix, hemi_prefix, output_path : str
        Path and naming parameters for saving files.
    """
    n_comm, N = A_thr.shape
    
    # Generate color list
    cmap = plt.get_cmap(cmap_name, n_comm)
    colors = [mcolors.to_hex(cmap(i)) for i in range(n_comm)]
    
    # Initialize RGBA array: (height, width, RGBA)
    # Default is white (1,1,1) with alpha 0 (fully transparent)
    rgb_array = np.ones((n_comm, N, 4))
    rgb_array[..., 3] = 0 
    
    for k in range(n_comm):
        mask = ~np.isnan(A_thr[k, :])
        # Set RGB
        rgb_array[k, mask, :3] = mcolors.to_rgb(colors[k])
        # Map proportion to alpha (scaled for visibility)
        rgb_array[k, mask, 3] = np.clip(A_thr[k, mask] * 1.5, 0, 1)
        
    fig, ax = plt.subplots(figsize=(16, 4))
    ax.imshow(rgb_array, aspect='auto', origin='upper')
    
    # Formatting
    ax.set_xlabel("Nodes (grouped by network)", fontsize=22)
    ax.set_ylabel("Edge communities", fontsize=22)
    ax.set_yticks(np.arange(n_comm))
    ax.set_yticklabels(np.arange(1, n_comm + 1), fontsize=18)
    ax.set_xticks([])
    ax.set_title(f"Community affiliation from {hemi_prefix} (>{threshold*100:.0f}%)", 
                 fontsize=28, pad=15)
    
    # Save figures
    if output_path:
        import os
        for fmt in ['png', 'eps']:
            save_dir = os.path.join(output_path, fmt)
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(save_dir, f"{prefix}_affiliation.{fmt}")
            fig.savefig(save_file, dpi=600, bbox_inches='tight', transparent=True)
            
    plt.tight_layout()
    plt.show()
    return fig








import seaborn as sns
def plot_box_and_jitter(
    df, 
    x_col='Community', 
    y_col='Value', 
    hue_col=None,  
    palette_name='PuBu', # Can be a string (Seaborn palette) or a dict (Mapping)
    box_width=0.5,
    xtricks_list=None, 
    yrange=None,
    output_path='.',
    prefix='plot',
    dataset_name='REST',
    title='',
    y_label='',
    x_label=None,
    jitter_w=0.1, 
    legend_loc='best',
    legend_bbox=None,
    figsize=(12, 4),
    show_legend=True,
    point_size=10,
    dpi=600
):
    """
    Generates a high-quality boxplot with manually overlaid jittered scatter points.
    
    Ensures color consistency by using a dictionary mapping for both boxes and points.
    All labels and structure are optimized for academic publication.
    """
    plt.figure(figsize=figsize)
    ax = plt.gca()

    # --- Step 1: Logic for hue and category ordering ---
    # Determine if we are grouping by an additional variable (hue)
    is_hue_active = (hue_col is not None and hue_col in df.columns)
    actual_hue = hue_col if is_hue_active else x_col
    
    # Force sort order to ensure X-axis and Colors stay aligned across manual plotting
    unique_x = xtricks_list
    unique_hue_groups = sorted(df[actual_hue].unique())

    # --- Step 2: Palette Processing ---
    # Convert palette name to a name-to-color mapping dictionary
    if isinstance(palette_name, dict):
        current_palette = palette_name
    else:
        # Generate palette colors from a standard seaborn name
        colors = sns.color_palette(palette_name, n_colors=len(unique_hue_groups))
        current_palette = dict(zip(unique_hue_groups, colors))

    # --- Step 3: Draw the Boxplot ---
    # Filter out zero values if they represent missing/non-existent data
    df_plot = df[df[y_col] != 0].copy()
    
    sns.boxplot(
        data=df_plot, 
        x=x_col, 
        y=y_col,
        order=unique_x,       # Explicit X-axis order
        hue=actual_hue,
        hue_order=unique_hue_groups,
        palette=current_palette, 
        width=box_width, 
        dodge=True if is_hue_active else False, 
        showfliers=False,     # Outliers will be handled by the scatter points
        zorder=1,
        ax=ax,
        legend=True,
        # boxprops=dict(alpha=0.3)
    )
    
    # --- Step 4: Draw Manual Jittered Scatter Plot ---
    # Calculate horizontal separation (dodge) for grouped data
    dodge_sep = (box_width / len(unique_hue_groups)) if is_hue_active else 0

    plot_stripplot_manual(
        df=df_plot, 
        x_col=x_col, 
        y_col=y_col, 
        hue_col=actual_hue, 
        palette=current_palette, # Pass the dictionary directly
        x_order=unique_x,
        hue_order=unique_hue_groups,
        dodge_separation=dodge_sep,
        jitter_width=jitter_w,
        ax=ax,
        s=point_size, 
        linewidth=0.3
    )

    # --- Step 5: Handle Legend ---
    if show_legend and is_hue_active:
        legend_kwargs = {'loc': legend_loc, 'title': None, 'fontsize': 12}
        if legend_bbox is not None:
            legend_kwargs['bbox_to_anchor'] = legend_bbox
        
        # Avoid duplicate labels in the legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[:len(unique_hue_groups)], labels[:len(unique_hue_groups)], **legend_kwargs)
    else:
        # Hide legend if not needed
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # --- Step 6: Aesthetics and Formatting ---
    ax.set_title(title, fontsize=18, pad=10)
    ax.set_xlabel(x_label if x_label is not None else x_col, fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)
    
    if yrange is not None: 
        ax.set_ylim(yrange)
        if yrange[0] <= 0 <= yrange[1]:
            num_ticks = 7  
        else:
            num_ticks = 5

        ticks = np.linspace(yrange[0], yrange[1], num_ticks)
        ax.set_yticks(ticks)
        
        if all(float(t).is_integer() for t in ticks):
            ax.set_yticklabels([f"{int(t)}" for t in ticks], fontsize=14)
        else:
            ax.set_yticklabels([f"{t:.1f}" for t in ticks], fontsize=14)
            
    if xtricks_list is not None:
        ax.set_xticklabels(xtricks_list, fontsize=16)
    
    ax.tick_params(axis='y', labelsize=14)
    
    # Add vertical dashed lines as visual separators between X categories
    for i in range(len(unique_x) - 1):
        ax.axvline(i + 0.5, color='lightgray', linestyle='--', linewidth=0.5)
            
    # Add horizontal baseline at Y=0 if the range spans zero
    if yrange is not None and yrange[0] < 0 and yrange[1] > 0:
        ax.axhline(0, color="gray", linestyle="--", linewidth=1, zorder=0)

    plt.tight_layout()

    # --- Step 7: Export Images ---
    filename = f"{dataset_name}_{prefix}"
    for fmt in ['png', 'eps']:
        save_dir = os.path.join(output_path, fmt)
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f"{save_dir}/{filename}.{fmt}", dpi=dpi, bbox_inches='tight')
    
    plt.show()
    return ax

def plot_stripplot_manual(
    df, x_col, y_col, hue_col, palette, 
    x_order, hue_order,
    dodge_separation=0, jitter_width=0.03, ax=None, **scatter_kwargs
):
    """
    Manually renders jittered scatter points aligned with the boxplot coordinates.
    Uses dictionary-based color mapping to ensure boxes and points match.
    """
    if ax is None: ax = plt.gca()
    
    is_single_mode = (x_col == hue_col)

    # Iterate through each X category (e.g., each Community)
    for i, net in enumerate(x_order):
        # Iterate through each Hue sub-group (e.g., Modality Types)
        sub_groups = [net] if is_single_mode else hue_order
        
        for j, grp in enumerate(sub_groups):
            # Filter data for specific X and Hue intersection
            mask = (df[x_col] == net) & (df[hue_col] == grp)
            vals = df.loc[mask, y_col].values
            
            if len(vals) == 0: continue

            # Apply random horizontal displacement (jitter)
            jitter = np.random.uniform(-jitter_width, jitter_width, size=len(vals))
            
            # Calculate final X-coordinate: 
            # Main index (i) + Group offset (Dodge) + Random Noise (Jitter)
            if is_single_mode:
                offset = 0
                color_val = palette[net] # Key-based color retrieval
            else:
                offset = (j - (len(hue_order) - 1) / 2) * dodge_separation
                color_val = palette[grp] # Key-based color retrieval

            # Plot raw data points
            ax.scatter(
                i + offset + jitter, 
                vals,
                color=color_val,
                edgecolors='#333333',
                zorder=2, # Ensure points are rendered above the boxes
                **scatter_kwargs
            )
    

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def plot_bar_by_modal(matrix_data, network_names, dataset, prefix, output_path):
    """
    Perform statistical analysis (ANOVA) and visualize differences between 
    connection modal types (U-U, H-H, H-U).
    
    Parameters:
    -----------
    matrix_data : ndarray
        N x N matrix containing the metric to be analyzed (e.g., connectivity or entropy).
    network_names : list
        List of network labels for each node in the matrix.
    dataset : str
        Name of the dataset (e.g., 'HCP').
    prefix : str
        Metric name used for labeling and filenames.
    output_path : str
        Base directory for saving plots.
    """
    
    # 1. Define network hierarchy mapping
    hetero_nets = ['L_DMN', 'L_FPN', 'L_LIMB', 'L_VAN', 'L_DAN',
                   'R_DMN', 'R_FPN', 'R_LIMB', 'R_VAN', 'R_DAN']
    
    net_to_type = {net: 'Hetero' if net in hetero_nets else 'Uni' for net in network_names}

    # 2. Extract edge-level data and assign connection types
    plot_data = []
    n_nodes = len(network_names)
    # Use lower triangle to avoid duplicate edges in symmetric matrices
    rows, cols = np.tril_indices(n_nodes, k=0)
    
    for r, c in zip(rows, cols):
        type_row = net_to_type[network_names[r]]
        type_col = net_to_type[network_names[c]]
        
        # Determine connection modality category
        if type_row == 'Hetero' and type_col == 'Hetero':
            conn_type = 'H - H'
        elif type_row == 'Uni' and type_col == 'Uni':
            conn_type = 'U - U'
        else:
            conn_type = 'H - U'
            
        plot_data.append({
            'Value': matrix_data[r, c],
            'Connection_Type': conn_type
        })
    
    df_plot = pd.DataFrame(plot_data)
    
    # 3. Statistical Analysis: ANOVA and Post-hoc Tukey HSD
    # Fitting the linear model for ANOVA
    model = ols('Value ~ C(Connection_Type)', data=df_plot).fit()
    anova_table = sm.stats.anova_lm(model, typ=3)
    
    # Pairwise comparisons to identify specific group differences
    tukey = pairwise_tukeyhsd(df_plot['Value'], df_plot['Connection_Type'], alpha=0.05)
    
    print("--- ANOVA Results ---")
    print(anova_table)
    print("\n--- Tukey HSD Post-hoc ---")
    print(tukey)

    # 4. Visualization: Combined Bar and Strip Plot
    plt.figure(figsize=(4, 6))
    
    # Render Bar plot for mean and error bars
    sns.barplot(
        data=df_plot, 
        x='Connection_Type', 
        y='Value', 
        palette='PuBu', 
        alpha=0.7, 
        err_kws={'linewidth': 2}
    )

    # Overlay Strip plot for individual data points distribution
    sns.stripplot(
        data=df_plot, 
        x='Connection_Type', 
        y='Value', 
        palette='PuBu',   
        alpha=0.6,       
        jitter=0.2,      
        size=4,
        linewidth=0.5,
        edgecolor='gray'
    )

    # Aesthetic refinements
    plt.ylabel(f'{prefix}', fontsize=20)
    plt.xlabel('Connection Modal Type', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    sns.despine() 

    # 5. Save outputs in multiple formats
    for fmt in ['png', 'eps']:
        save_dir = os.path.join(output_path, fmt)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{dataset}_{prefix}_modal_all.{fmt}')
        plt.savefig(save_path, dpi=600, bbox_inches='tight')
    
    plt.show()
    


def plot_consumbar_by_modal(matrix_data, 
                            network_names, 
                            custom_colors, 
                            dataset, 
                            prefix, 
                            output_path):
    """
    Visualize the composition of edge communities within each modality connection type 
    (H-H, U-U, H-U) using a stacked bar chart.

    Scientific Significance:
    -----------------------
    Reveals how different edge communities are distributed across the brain's 
    hierarchical levels (unimodal vs. heteromodal).
    """
    
    # 1. Define network hierarchy mapping
    hetero_nets = ['L_DMN', 'L_FPN', 'L_LIMB', 'L_VAN', 'L_DAN',
                   'R_DMN', 'R_FPN', 'R_LIMB', 'R_VAN', 'R_DAN']
    net_to_type = {net: 'H' if net in hetero_nets else 'U' for net in network_names}

    # 2. Extract edge community assignments and their modal groups
    data_list = []
    n_nodes = len(network_names)
    # Use lower triangle excluding diagonal (k=-1) to process unique edges
    rows, cols = np.tril_indices(n_nodes, k=-1)
    
    for r, c in zip(rows, cols):
        type_row = net_to_type[network_names[r]]
        type_col = net_to_type[network_names[c]]

        # Categorize the edge modality
        if type_row == 'H' and type_col == 'H':
            group = 'H - H'
        elif type_row == 'U' and type_col == 'U':
            group = 'U - U'
        else:
            group = 'H - U'
            
        # Get the community ID assigned to this edge
        comm_id = matrix_data[r, c]
        
        data_list.append({
            'Group': group,
            'Community': comm_id
        })
    
    df_comm = pd.DataFrame(data_list)
    
    # Ensure Community is treated as a categorical string for proper hue mapping
    df_comm['Community'] = df_comm['Community'].astype(int).astype(str)
    # Define order to ensure legend/stacking follows 1, 2, 3...
    hue_order = [str(i) for i in sorted(df_comm['Community'].unique().astype(int))]

    # 3. Visualization: Stacked Histogram (Composition Bar)
    f, ax = plt.subplots(figsize=(5, 6))
    
    sns.histplot(
        data=df_comm,
        x='Group',
        hue='Community',
        hue_order=hue_order,
        multiple='stack',      # Core: stack bars to show composition
        palette=custom_colors,
        shrink=0.7,            # Adjust bar width
        edgecolor='white',     # Distinct boundaries between segments
        linewidth=1,
        legend=False,          # Set to True if legend is needed
        ax=ax
    )

    # 4. Aesthetic Refinements
    sns.despine() 
    ax.set_xlabel('Connection Modal Type', fontsize=24)
    ax.set_ylabel('Edge Count', fontsize=24)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    # 5. Save Outputs
    plt.tight_layout()
    if output_path:
        for fmt in ['png', 'eps']:
            save_dir = os.path.join(output_path, fmt)
            os.makedirs(save_dir, exist_ok=True)
            save_file = os.path.join(save_dir, f'{dataset}_{prefix}_modality_composition.{fmt}')
            plt.savefig(save_file, dpi=600, bbox_inches='tight')
            
    plt.show()
    

def plot_comprehensive_coupling(key, 
                                title_name, 
                                LI_coupling, 
                                output_path, 
                                custom_colors, 
                                plot_mode='strength',
                                y_limit=(-0.5,0.5)):
    """
    Comprehensive visualization comparing community-level coupling with whole-brain levels.
    
    Scientific Significance:
    -----------------------
    Determines if specific edge communities exhibit significantly higher or lower 
    integration/segregation coupling compared to the global (whole-brain) average.

    Parameters:
    -----------
    key : str
        'inte' or 'segre' corresponding to the dictionary key in coupling_data.
    title_name : str
        Display name for the metric ('integration' or 'segregation').
    coupling_data : dict
        Nested dictionary containing community and whole-brain coupling metrics.
    output_path : str
        Directory to save the resulting plots.
    custom_colors : list
        Color list for the communities.
    plot_mode : str
        'mean' for average coupling values or 'slope' for coupling sensitivity (slope).
    """

    # 1. Dynamic community label generation based on data dimensions
    n_comm = len(LI_coupling['inte']['slope_pair_comm'])

    communities = [f'C{i}' for i in range(1, n_comm + 1)]    
    communities_wb = np.hstack((communities, 'WB'))    

    
    # 2. Data Selection: Mean-based vs. Slope-based analysis
    if plot_mode == 'strength':
        # (n_comm, n_net, n_subjects) -> mean across networks
        raw_data = [np.nanmean(data, axis=1) for data in LI_coupling[key]['strength_pair_comm']]
        wb_values = np.nanmean(LI_coupling[key]['strength_pair_allcomm'][0], axis=1)
        y_label_text = f"Mean of {title_name}"
        save_suffix = "mean_all_community"

    else:
        # Direct extraction of pre-calculated slope values
        raw_data = LI_coupling[key]['slope_pair_comm'] 
        wb_values = LI_coupling[key]['slope_pair_allcomm'][0]
        y_label_text = f"Slope of {title_name}"
        save_suffix = "slope_all_community"


    # 3. Data Reshaping: Transform to long-format for Seaborn-style plotting
    df_long = pd.DataFrame(raw_data).T.melt(var_name="community", value_name="value")
    
    # Matching community names with data points (assuming 485 subjects/samples)
    num_samples = raw_data.shape[1] if plot_mode == 'mean' else len(raw_data) 
    df_long['community'] = np.concatenate([np.repeat(c, 485) for c in communities])

    # 4. Global Baseline: Prepare whole-brain data for comparison
    df_wb = pd.DataFrame({
        'value': wb_values,
        'community': 'WB'
    })

    # Combine community data and whole-brain baseline
    df_comb = pd.concat([df_long, df_wb], ignore_index=True)

    # 5. Visual Palette Setup: Add gray color for the whole-brain baseline
    custom_colors_wb = np.hstack((custom_colors, '#808080'))
    palette = {c: col for c, col in zip(sorted(df_comb['community'].unique()), custom_colors_wb)}

    # 6. Execution: Delegate to the specialized box-jitter plotting function
    plot_box_and_jitter(
        df=df_comb, 
        x_col='community', 
        y_col='value', 
        palette_name=palette,
        box_width=0.5,
        xtricks_list=communities_wb,
        yrange=y_limit,
        output_path=output_path,
        prefix=f"{key}-coupling_{save_suffix}",
        dataset_name='REST',
        title=f"{plot_mode.capitalize()} of {title_name} coupling",
        y_label=y_label_text,
        x_label='Communities ID',
        jitter_w=0.1,
        legend_loc='best',
        figsize=(12, 4),
        point_size=3,
        dpi=600
    )


def plot_joint_correlation(df, x, y, x_lim, y_lim, output_path, filename):
    """
    Generate a joint plot with automatic Spearman correlation 
    formatting for academic publication.
    """
    # --- 1. Statistical Calculation ---
    rho, p = spearmanr(df[x], df[y],nan_policy='omit')
    
    # --- 2. Professional Label Formatting ---
    rho_text = f"rho = {rho:.2f}"
    if p < 0.001:
        # Extract scientific notation components
        base, exponent = f"{p:.2e}".split('e')
        # Clean up exponent (remove leading zeros)
        p_text = fr"$p = {base} \times 10^{{{int(exponent)}}}$"
    else:
        p_text = fr"$p = {p:.3f}$"
    
    stats_label = f"{rho_text}\n{p_text}"

    # --- 3. Visualization ---
    # Using 'reg' kind for regression line and confidence interval
    g = sns.jointplot(data=df, x=x, y=y, kind='reg', 
                      height=5, marginal_kws=dict(bins=10, fill=True),
                      scatter_kws={'s': 40, 'alpha': 0.6, 'edgecolor': 'white'},
       
                      )
    
    # Position the text label in the upper left corner
    g.ax_joint.text(
        0.02, 0.98, stats_label,
        ha='left', va='top',
        transform=g.ax_joint.transAxes,
        fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )

    # Standardize axes and labels
    g.ax_joint.tick_params(labelsize=10)
    # if not np.isnan(x_lim):
    g.ax_joint.set_xlim(x_lim)
    g.ax_joint.set_ylim(y_lim) # Adjusted for consistency
    g.ax_joint.set_xlabel(x.replace('_', ' ').capitalize(), fontsize=10)
    g.ax_joint.set_ylabel(y.replace('_', ' ').capitalize(), fontsize=10)

    # --- 4. High-Resolution Export ---
    for fmt in ['png', 'eps']:
        save_dir = os.path.join(output_path, fmt)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        g.fig.savefig(f'{save_dir}/{filename}.{fmt}', dpi=600, bbox_inches='tight')
    
    return g

def analyze_node_profiles(ci_matrix, 
                          labels_dict, 
                          network_names_yeo7, 
                          output_path, 
                          prefix, 
                          custom_cmap,
                          plot_figure=True):
    """
    Computes node-level community metrics (entropy, affiliation) for intra- and 
    inter-hemispheric connectivity matrices.
    
    Parameters:
    -----------
    ci_matrix : ndarray
        The full community index/connectivity matrix (e.g., 400x400).
    labels_dict : dict
        Dictionary containing network labels for L/R cortex.
    network_names_yeo7 : list
        List of names for the 7 Yeo networks.
    """
    
    n_nodes = ci_matrix.shape[0]
    mid = n_nodes // 2  # Split point for hemispheres (e.g., 200 for a 400-node atlas)
    
    # Pre-define triangular mask for square (intra-hemispheric) matrices
    tri_mask_hemi = np.triu(np.ones(mid, dtype=bool), 1)
    
    # Configuration map: {suffix: (row_slice, col_slice, label_key)}
    # Note: For LR/RL, we determine which hemisphere's labels to use based on logic
    configs = {
        'LL': (slice(0, mid), slice(0, mid), 'L_cortex'),
        'RR': (slice(mid, n_nodes), slice(mid, n_nodes), 'R_cortex'),
        'LR': (slice(0, mid), slice(mid, n_nodes), 'R_cortex'),
        'RL': (slice(mid, n_nodes), slice(0, mid), 'L_cortex')
    }

    results = {}

    for suffix, (row_s, col_s, lab_key) in configs.items():
        # Extract sub-matrix for the specific hemispheric combination
        sub_matrix = ci_matrix[row_s, col_s]
        
        # Determine CI input: 
        # Intra-hemispheric (LL/RR) are symmetric, use upper triangle.
        # Inter-hemispheric (LR/RL) are typically not symmetric, use flatten.
        if suffix in ['LL', 'RR']:
            input_ci = sub_matrix[tri_mask_hemi]
        else:
            # Flatten is used for rectangular/asymmetric inter-hemispheric blocks
            input_ci = sub_matrix.flatten()

        # Execute the core analysis function
        results[suffix] = calc_edge_communities_to_nodes(
            ci=input_ci,
            lab=labels_dict['yeo7'][lab_key],
            net=network_names_yeo7, 
            threshold=0.2,
            cmap=custom_cmap,
            output_path=output_path,
            prefix=f'{prefix}_{suffix}',
            hemi_prefix=suffix,
            plot_figure=plot_figure
        )

    return results

from joblib import Parallel, delayed
from tqdm import tqdm
def analyze_batch_null_node_profiles(
    ci_matrix_null,
    labels_dict,
    network_names,
    output_path,
    prefix,
    custom_cmap,
    n_jobs=8
):
    """
    Execute the hemispheric distribution pipeline on null data in parallel.
    This generates a distribution of node metrics (entropy/affiliation) 
    to establish statistical significance.

    Parameters:
    -----------
    ci_matrix_null : ndarray
        3D array of shape (n_nodes, n_nodes, n_perm).
    n_jobs : int
        Number of parallel CPU cores to use.
    """
    n_perm = ci_matrix_null.shape[-1]

    # Local helper function for parallel execution
    def _process_permutation(n):
        # Extract the n-th null community matrix
        null_ci_instance = ci_matrix_null[:, :, n]
        
        # We reuse the top-level analyzer (the one we named earlier)
        # Note: plot_figure is False to save time and I/O during null runs
        return analyze_node_profiles(
            ci_matrix=null_ci_instance,
            labels_dict=labels_dict,
            network_names_yeo7=network_names,
            output_path=output_path,
            prefix=f"{prefix}_null{n}",
            custom_cmap=custom_cmap,
            plot_figure=False 
        )

    # Parallel computation with progress bar
    with Parallel(n_jobs=n_jobs, backend='loky') as parallel:
        null_results = parallel(
            delayed(_process_permutation)(n)
            for n in tqdm(range(n_perm), desc="Batch analyze null metrics", disable=False)
        )

    return null_results
from statsmodels.stats.multitest import fdrcorrection
def analyze_network_alliliation_profiles(
    community_node_affi_all,
    community_node_affi_all_null,
    labels_dict,
    net_yeo7,
    community_colors,
    output_path,
    dataset='REST'
):
    """
    Compute network-level integration / segregation significance 
    for each community using a null distribution and save null data.

    Parameters
    ----------
    community_node_affi_all : dict
        {"LL","RR","LR","RL"} (5 community, 200 nodes)
    community_node_affi_all_null : dict
        {"LL","RR","LR","RL"} (n_perm, 5 community, 200 nodes)
    labels_dict : dict
        network label mapping (L & R hemispheres)
    net_yeo7 : list of str
        Yeo-7 network names
    output_path : str
        Folder for figure/data output
    dataset : str
        Dataset identifier for filename

    Returns
    -------
    dict with p-values, significance matrices, and path to saved null distributions.
    """
    
    # -----------------------------------------------
    # Basic info
    # -----------------------------------------------
    n_perm = community_node_affi_all_null["LL"].shape[0]
    n_comm = community_node_affi_all["LL"].shape[0]
    n_network = len(net_yeo7)
    comm_names = [f"C{i+1}" for i in range(n_comm)]

    # Matrices to store results
    p_inte_matrix = np.zeros((n_comm, n_network))
    p_segre_matrix = np.zeros((n_comm, n_network))
    mean_inte_matrix = np.zeros((n_comm, n_network))
    mean_segre_matrix = np.zeros((n_comm, n_network))

    # label vectors
    labels_L = labels_dict["yeo7"]["L_cortex"]
    labels_R = labels_dict["yeo7"]["R_cortex"]

    inte_dfs = []
    segre_dfs = []
    all_null_inte_dist = []  
    all_null_segre_dist = [] 

    # ===========================================================
    # LOOP: each community
    # ===========================================================
    for icom in tqdm(range(n_comm), desc="Communities"):

        # --- Real Data Processing ---
        LL = community_node_affi_all["LL"][icom]
        RR = community_node_affi_all["RR"][icom]
        LR = community_node_affi_all["LR"][icom]
        RL = community_node_affi_all["RL"][icom]
        
        # for arr in [LL,RR,LR,RL]:
        #     arr[arr==0] = np.nan
        
        # Aggregate node values to network level
        LL_net, df_LL = aggregate_node_to_network(LL, labels_L, net_yeo7)
        RR_net, df_RR = aggregate_node_to_network(RR, labels_R, net_yeo7)
        LR_net, df_LR = aggregate_node_to_network(LR, labels_R, net_yeo7)
        RL_net, df_RL = aggregate_node_to_network(RL, labels_L, net_yeo7)

        # Extract network means (Real values for this community)
        rLL, rRR, rLR, rRL = [(net['fc_network_mean']) for net in [LL_net, RR_net, LR_net, RL_net]]
        
        # Calculate real integration/segregation means per network
        # These are the "Real Means" you see in the summary
        real_inte = (rLL + rLR) - (rRR + rRL)
        real_segre = (rLL - rLR) - (rRR - rRL)
        
        mean_inte_matrix[icom] = real_inte
        mean_segre_matrix[icom] = real_segre

        # Node-wise df (for boxplots/distributions)
        # We ensure numeric selection and alignment
        val_LL = df_LL.select_dtypes("number")["value"].values
        val_RR = df_RR.select_dtypes("number")["value"].values
        val_LR = df_LR.select_dtypes("number")["value"].values
        val_RL = df_RL.select_dtypes("number")["value"].values

        df_inte = pd.DataFrame({"value": (val_LL + val_LR) - (val_RR + val_RL)})
        df_segre = pd.DataFrame({"value": (val_LL - val_LR) - (val_RR - val_RL)})

        for df_tmp in [df_inte, df_segre]:
            df_tmp["value"] = df_tmp["value"].fillna(0)
            df_tmp["network"] = df_LL["network"].values
            df_tmp["community"] = comm_names[icom]

        inte_dfs.append(df_inte)
        segre_dfs.append(df_segre)

        # --- Null Distribution Calculation ---
        null_inte = np.zeros((n_perm, n_network))
        null_segre = np.zeros((n_perm, n_network))

        for p in range(n_perm):
            LL_n = community_node_affi_all_null["LL"][p, icom]
            RR_n = community_node_affi_all_null["RR"][p, icom]
            LR_n = community_node_affi_all_null["LR"][p, icom]
            RL_n = community_node_affi_all_null["RL"][p, icom]
            
            # for arr in [LL_n,RR_n,LR_n,RL_n]:
            #     arr[arr==0] = np.nan
            
            # Simplified aggregation for speed inside loop
            n_LL, _ = aggregate_node_to_network(LL_n, labels_L, net_yeo7)
            n_RR, _ = aggregate_node_to_network(RR_n, labels_R, net_yeo7)
            n_LR, _ = aggregate_node_to_network(LR_n, labels_R, net_yeo7)
            n_RL, _ = aggregate_node_to_network(RL_n, labels_L, net_yeo7)

            v_LL, v_RR, v_LR, v_RL = [(n['fc_network_mean']) for n in [n_LL, n_RR, n_LR, n_RL]]

            null_inte[p] = (v_LL + v_LR) - (v_RR + v_RL)
            null_segre[p] = (v_LL - v_LR) - (v_RR - v_RL)

        all_null_inte_dist.append(null_inte)
        all_null_segre_dist.append(null_segre)

        # p-value calculation
        # NOTE: This p-value is testing against the Null Distribution's actual spread
        # Not just the distance from 0.
        for inet in range(n_network):
            # Integration P
            obs_i = real_inte[inet]
            null_i = null_inte[:, inet]
            p_inte_matrix[icom, inet] = (np.sum(np.abs(null_i - np.mean(null_i)) >= np.abs(obs_i - np.mean(null_i))) + 1) / (n_perm + 1)
            
            # Segregation P
            obs_s = real_segre[inet]
            null_s = null_segre[:, inet]
            # ÕâÀïÎÒ¸ÄÎª¸üÎÈ½¡µÄ£º¹Û²âÖµÔÚNull·Ö²¼ÖÐµÄ°Ù·ÖÎ»ÊýÎ»ÖÃ (Ë«Î²)
            p_left = (np.sum(null_s <= obs_s) + 1) / (n_perm + 1)
            p_right = (np.sum(null_s >= obs_s) + 1) / (n_perm + 1)
            p_segre_matrix[icom, inet] = 2 * min(p_left, p_right)

    # ===========================================================
    # Final Significance & DataFrame Conversion
    # ===========================================================
    def to_df(matrix):
        return pd.DataFrame(matrix, index=comm_names, columns=net_yeo7)

    # FDR Correction
    _, p_adj_inte = fdrcorrection(p_inte_matrix.flatten(), 0.05)
    sig_inte_df = to_df(p_adj_inte.reshape((n_comm, n_network)))
    
    _, p_adj_segre = fdrcorrection(p_segre_matrix.flatten(), 0.05)
    sig_segre_df = to_df(p_adj_segre.reshape((n_comm, n_network)))

    # Mean Matrices to DF
    mean_inte_df = to_df(mean_inte_matrix)
    mean_segre_df = to_df(mean_segre_matrix)
    
    # Raw P-values to DF
    p_inte_df = to_df(p_inte_matrix)
    p_segre_df = to_df(p_segre_matrix)

    # Save to file
    os.makedirs(output_path, exist_ok=True)
    save_filename = os.path.join(output_path, f"{dataset}_network_affiliation_results.npz")
    np.savez(save_filename, 
             null_inte_all=np.array(all_null_inte_dist),
             null_segre_all=np.array(all_null_segre_dist),
             mean_inte=mean_inte_matrix,
             mean_segre=mean_segre_matrix,
             p_inte=p_inte_matrix,
             p_segre=p_segre_matrix)

    return {
        "df_all_inte": pd.concat(inte_dfs, ignore_index=True),
        "df_all_segre": pd.concat(segre_dfs, ignore_index=True),
        "mean_inte": mean_inte_df,    # <--- Added
        "mean_segre": mean_segre_df,  # <--- Added
        "p_inte": p_inte_df,
        "p_segre": p_segre_df,
        "sig_inte": sig_inte_df,      # Now DataFrame
        "sig_segre": sig_segre_df,    # Now DataFrame
        "null_dist_file": save_filename
    }
