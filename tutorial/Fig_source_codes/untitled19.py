#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined MERSCOPE analysis (final version):
- Load & process SD & Chenxi models (keep only 'ctl' condition)
- Classify cells by SOX3.S (≥9) and SOX17A.S (≥5)
- Violin plot: cell volume by cell type and model
- Spatial plots: SOX3, SOX17A expression & cell type classification
- Cumulative volume distribution (1k–30k) with crossover marker
- SOX17 expression vs volume scatter plots + statistics
"""

import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats   # <-- for Mann-Whitney U test

# ----------------------------------------------------------------------
# 1. Load HDF5 file (MERSCOPE format)
# ----------------------------------------------------------------------
def load_merscope_hdf5(file_path):
    """Load your specific MERSCOPE HDF5 structure."""
    with h5py.File(file_path, 'r') as f:
        X = np.array(f['X'][:])
        print(f"Expression matrix shape: {X.shape}")

        obs_data = {}
        for key in f['obs'].keys():
            if key not in ['__categories', '_index']:
                obs_data[key] = np.array(f['obs'][key])

        var_index = np.array(f['var']['_index'][:]).astype(str)

        if 'spatial' in f['obsm']:
            spatial_coords = np.array(f['obsm']['spatial'][:])
        elif 'center_x' in obs_data and 'center_y' in obs_data:
            spatial_coords = np.column_stack([obs_data['center_x'],
                                              obs_data['center_y']])
        else:
            spatial_coords = None

        obs_df = pd.DataFrame(obs_data)
        if '_index' in f['obs']:
            obs_df.index = np.array(f['obs']['_index'][:]).astype(str)

        return {
            'expression': X,
            'obs': obs_df,
            'var': var_index,
            'spatial': spatial_coords
        }

# ----------------------------------------------------------------------
# 2. Process a single model: filter, annotate, assign cell types
# ----------------------------------------------------------------------
def process_model(file_path, annotations_path, model_name):
    """
    Load data, merge annotations, keep only 'ctl' condition,
    classify cells by SOX3.S and SOX17A.S expression.
    Returns:
        data : dict with expression, obs, spatial, var
        result_df : DataFrame with cell_type, volume, model, expression values
    """
    data = load_merscope_hdf5(file_path)
    annotations = pd.read_csv(annotations_path)

    data['obs']['EntityID'] = data['obs'].index

    data['obs'] = data['obs'].merge(
        annotations[['Custom cell groups', 'stage', 'condition']],
        on='Custom cell groups',
        how='left'
    )

    # Remove rows with missing stage/condition
    valid_mask = data['obs']['stage'].notna() & data['obs']['condition'].notna()
    data['obs'] = data['obs'][valid_mask]
    data['expression'] = data['expression'][valid_mask]
    data['spatial'] = data['spatial'][valid_mask]

    # Keep only 'ctl' condition
    ctl_mask = data['obs']['condition'] == 'ctl'
    data['obs'] = data['obs'][ctl_mask]
    data['expression'] = data['expression'][ctl_mask]
    data['spatial'] = data['spatial'][ctl_mask]

    expression_matrix = data['expression']
    var_names = data['var']
    obs_data = data['obs']

    # Gene indices
    sox3_idx = np.where(var_names == 'SOX3.S')[0][0]
    sox17a_idx = np.where(var_names == 'SOX17A.S')[0][0]

    sox3_exp = expression_matrix[:, sox3_idx]
    sox17a_exp = expression_matrix[:, sox17a_idx]

    sox3_th = 9
    sox17a_th = 5

    sox3_pos = sox3_exp >= sox3_th
    sox17a_pos = sox17a_exp >= sox17a_th
    double_pos = sox3_pos & sox17a_pos

    cell_types = []
    volumes = []
    for i in range(len(sox3_exp)):
        vol = obs_data.iloc[i]['volume'] if 'volume' in obs_data.columns else np.nan
        if double_pos[i]:
            ct = 'Double Positive'
        elif sox3_pos[i]:
            ct = 'SOX3+ (Ectoderm)'
        elif sox17a_pos[i]:
            ct = 'SOX17A+ (Endoderm)'
        else:
            ct = 'Negative'
        cell_types.append(ct)
        volumes.append(vol)

    # Add columns to obs for easier plotting
    obs_data['cell_type'] = cell_types
    obs_data['sox3_expression'] = sox3_exp
    obs_data['sox17a_expression'] = sox17a_exp
    obs_data['model'] = model_name

    # Result DataFrame for volume comparison
    result_df = pd.DataFrame({
        'cell_type': cell_types,
        'volume': volumes,
        'model': model_name,
        'sox3_expression': sox3_exp,
        'sox17a_expression': sox17a_exp
    })

    return data, result_df

# ----------------------------------------------------------------------
# 3. Plotting functions
# ----------------------------------------------------------------------
def plot_spatial_sox3(data, model_name):
    spatial = data['spatial']
    sox3_exp = data['obs']['sox3_expression']
    plt.figure(figsize=(10, 8))
    plt.scatter(spatial[:, 0], spatial[:, 1], c=sox3_exp,
                cmap='viridis', s=2, alpha=0.8, vmax=10)
    plt.colorbar(label='SOX3.S expression')
    plt.title(f'{model_name}: Spatial distribution of SOX3.S')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.tight_layout()
    plt.show()

def plot_spatial_sox17a(data, model_name):
    spatial = data['spatial']
    sox17a_exp = data['obs']['sox17a_expression']
    plt.figure(figsize=(10, 8))
    plt.scatter(spatial[:, 0], spatial[:, 1], c=sox17a_exp,
                cmap='viridis', s=2, alpha=0.8, vmax=10)
    plt.colorbar(label='SOX17A.S expression')
    plt.title(f'{model_name}: Spatial distribution of SOX17A.S')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.tight_layout()
    plt.show()

def plot_spatial_cell_types(data, model_name):
    spatial = data['spatial']
    obs = data['obs']
    colors = np.array(['lightgrey'] * len(obs))
    sox3_pos = obs['cell_type'] == 'SOX3+ (Ectoderm)'
    sox17a_pos = obs['cell_type'] == 'SOX17A+ (Endoderm)'
    double_pos = obs['cell_type'] == 'Double Positive'
    colors[sox3_pos] = 'blue'
    colors[sox17a_pos] = 'red'
    colors[double_pos] = 'black'
    sizes = np.ones(len(obs)) * 1
    sizes[sox3_pos | sox17a_pos | double_pos] = 3

    plt.figure(figsize=(12, 10))
    plt.scatter(spatial[:, 0], spatial[:, 1], c=colors, s=sizes, alpha=0.8)
    plt.title(f'{model_name}: Spatial distribution of cell types')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
                   markersize=8, label='SOX3+ only (ectoderm)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markersize=8, label='SOX17A+ only (endoderm)'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   markersize=8, label='Double positive'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgrey',
                   markersize=8, label='Negative')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_cumulative_volumes(obs_sd, obs_chenxi,
                            lower=1000, upper=30000,
                            label_sd='SD', label_chenxi='Chenxi'):
    """
    Plot cumulative cell counts by volume within a specified range.
    obs_sd, obs_chenxi : DataFrames (must have 'volume' column)
    """
    # Filter to the desired volume range
    sd_clean = obs_sd[(obs_sd['volume'] >= lower) & (obs_sd['volume'] <= upper)].copy()
    chenxi_clean = obs_chenxi[(obs_chenxi['volume'] >= lower) & (obs_chenxi['volume'] <= upper)].copy()

    print(f"{label_chenxi} cells in range: {len(chenxi_clean):,}")
    print(f"{label_sd} cells in range:     {len(sd_clean):,}")

    volume_thresholds = np.linspace(lower, upper, 200)
    chenxi_cum = [np.sum(chenxi_clean['volume'] <= t) for t in volume_thresholds]
    sd_cum = [np.sum(sd_clean['volume'] <= t) for t in volume_thresholds]

    plt.figure(figsize=(12, 8))
    plt.plot(volume_thresholds, chenxi_cum, color='blue', linewidth=3,
             label=f'{label_chenxi} (n={len(chenxi_clean):,})')
    plt.plot(volume_thresholds, sd_cum, color='orange', linewidth=3,
             label=f'{label_sd} (n={len(sd_clean):,})')

    # Find and mark crossover point where SD starts having more cells
    crossover_found = False
    for i, (c, s) in enumerate(zip(chenxi_cum, sd_cum)):
        if s > c and not crossover_found and volume_thresholds[i] > 2000:
            crossover_vol = volume_thresholds[i]
            plt.axvline(x=crossover_vol, color='red', linestyle='--',
                        label=f'SD Advantage Starts: {crossover_vol:.0f} volume')
            crossover_found = True

    plt.xlabel('Cell Volume Threshold')
    plt.ylabel('Cumulative Cell Count')
    plt.title(f'SD Model Shows Clear Advantage for Larger Cells\n({lower:,}–{upper:,} Volume Range)',
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ----------------------------------------------------------------------
# 4. SOX17 vs Volume scatter plots + statistics
# ----------------------------------------------------------------------
def plot_sox17_vs_volume(obs_sd, obs_chenxi,
                         sox17_threshold=5,
                         vol_lower=500, vol_upper=30000,
                         label_sd='SD', label_chenxi='Chenxi'):
    sd_clean = obs_sd[(obs_sd['volume'] >= vol_lower) & (obs_sd['volume'] <= vol_upper)].copy()
    chenxi_clean = obs_chenxi[(obs_chenxi['volume'] >= vol_lower) & (obs_chenxi['volume'] <= vol_upper)].copy()

    chenxi_sox17 = chenxi_clean['sox17a_expression']
    sd_sox17 = sd_clean['sox17a_expression']
    chenxi_volumes = chenxi_clean['volume']
    sd_volumes = sd_clean['volume']

    # ---- 1. Main scatter (expression vs volume) ----
    #plt.figure(figsize=(12, 8))
   # plt.scatter(chenxi_sox17, chenxi_volumes,
    #            alpha=0.6, s=30, color='blue', label=f'{label_chenxi} Model',
      #          edgecolors='white', linewidth=0.5)
    #plt.scatter(sd_sox17, sd_volumes,
     #           alpha=0.6, s=30, color='gold', label=f'{label_sd} Model',
       #         edgecolors='black', linewidth=0.5)
    #plt.axvline(x=sox17_threshold, color='red', linestyle='--', linewidth=2,
      #          label=f'SOX17+ Threshold (≥{sox17_threshold} transcripts)')
    #plt.xlabel('SOX17 Transcript Count', fontsize=12)
    #plt.ylabel('Cell Volume', fontsize=12)
    #plt.title('SOX17 Expression vs Cell Volume by Segmentation Model', fontsize=14, fontweight='bold')
    

    chenxi_pos = np.sum(chenxi_sox17 >= sox17_threshold)
    sd_pos = np.sum(sd_sox17 >= sox17_threshold)
    '''plt.text(0.02, 0.98,
             f'{label_chenxi} SOX17+ cells: {chenxi_pos}/{len(chenxi_sox17)} ({chenxi_pos/len(chenxi_sox17)*100:.1f}%)',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    plt.text(0.02, 0.88,
             f'{label_sd} SOX17+ cells: {sd_pos}/{len(sd_sox17)} ({sd_pos/len(sd_sox17)*100:.1f}%)',
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()'''

    # ---- 2. Focused scatter (SOX17+ only, same axes) ----
    chenxi_pos_mask = chenxi_sox17 >= sox17_threshold
    sd_pos_mask = sd_sox17 >= sox17_threshold

    '''plt.figure(figsize=(12, 8))
    plt.scatter(chenxi_sox17[chenxi_pos_mask], chenxi_volumes[chenxi_pos_mask],
                alpha=0.7, s=40, color='blue',
                label=f'{label_chenxi} SOX17+ (n={np.sum(chenxi_pos_mask)})',
                edgecolors='darkblue', linewidth=0.8)
    plt.scatter(sd_sox17[sd_pos_mask], sd_volumes[sd_pos_mask],
                alpha=0.7, s=40, color='gold',
                label=f'{label_sd} SOX17+ (n={np.sum(sd_pos_mask)})',
                edgecolors='darkorange', linewidth=0.8)
    plt.axvline(x=sox17_threshold, color='red', linestyle='--', linewidth=2)
    plt.xlabel('SOX17 Transcript Count', fontsize=12)
    plt.ylabel('Cell Volume', fontsize=12)
    plt.title('SOX17+ Endoderm Cells: Volume Distribution by Model', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()'''

    # ---- 3. Axes-exchanged scatter (Volume on X, SOX17 on Y) ----
    plt.figure(figsize=(14, 10))
    plt.scatter(sd_volumes[sd_pos_mask], sd_sox17[sd_pos_mask],
                alpha=0.7, s=40, color='gold',
                label=f'{label_sd} SOX17+ (n={np.sum(sd_pos_mask):,})',
                edgecolors='darkorange', linewidth=0.8)
    plt.scatter(chenxi_volumes[chenxi_pos_mask], chenxi_sox17[chenxi_pos_mask],
                alpha=0.7, s=40, color='blue',
                label=f'{label_chenxi} SOX17+ (n={np.sum(chenxi_pos_mask):,})',
                edgecolors='darkblue', linewidth=0.8)
    plt.axhline(y=sox17_threshold, color='red', linestyle='--', linewidth=2,
                label=f'SOX17+ Threshold (≥{sox17_threshold} transcripts)')
    plt.xlabel('Cell Volume', fontsize=14)
    plt.ylabel('SOX17 Transcript Count', fontsize=14)
    plt.title(f'SOX17+ Endoderm Cells: Expression vs Volume Distribution by Model\n'
              f'(Volume Range: {vol_lower:,}–{vol_upper:,})',
              fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---- Volume category analysis ----
    print("\n" + "="*60)
    print("SOX17+ CELLS ANALYSIS BY VOLUME CATEGORY")
    print("="*60)
    volume_categories = {
        'Small/Medium (500–5k)': (500, 5000),
        'Large (5k–10k)': (5000, 10000),
        'Very Large (>10k)': (10000, 20000)
    }
    for category, (vmin, vmax) in volume_categories.items():
        print(f"\n--- {category} ---")
        chenxi_mask = (chenxi_volumes >= vmin) & (chenxi_volumes < vmax)
        chenxi_total = np.sum(chenxi_mask)
        chenxi_pos = np.sum((chenxi_mask) & (chenxi_sox17 >= sox17_threshold))
        sd_mask = (sd_volumes >= vmin) & (sd_volumes < vmax)
        sd_total = np.sum(sd_mask)
        sd_pos = np.sum((sd_mask) & (sd_sox17 >= sox17_threshold))
        print(f"{label_chenxi}: {chenxi_pos}/{chenxi_total} SOX17+ cells ({chenxi_pos/max(chenxi_total,1)*100:.1f}%)")
        print(f"{label_sd}:     {sd_pos}/{sd_total} SOX17+ cells ({sd_pos/max(sd_total,1)*100:.1f}%)")
        if chenxi_pos > 0:
            print(f"Ratio (SD/{label_chenxi}): {sd_pos/chenxi_pos:.1f}x")

    # ---- Statistical comparison ----
    if np.sum(chenxi_pos_mask) > 0 and np.sum(sd_pos_mask) > 0:
        chenxi_vol_pos = chenxi_volumes[chenxi_pos_mask]
        sd_vol_pos = sd_volumes[sd_pos_mask]
        print(f"\nSOX17+ CELLS VOLUME COMPARISON:")
        print(f"{label_chenxi} SOX17+ cells: n={len(chenxi_vol_pos)}, mean volume={chenxi_vol_pos.mean():.1f}")
        print(f"{label_sd} SOX17+ cells: n={len(sd_vol_pos)}, mean volume={sd_vol_pos.mean():.1f}")
        stat, p_value = stats.mannwhitneyu(chenxi_vol_pos, sd_vol_pos, alternative='two-sided')
        print(f"Mann-Whitney U test: U={stat:.1f}, p={p_value:.3f}")
    else:
        print("\nNot enough SOX17+ cells for statistical comparison.")
def print_cell_type_summary(obs, model_name):
    counts = obs['cell_type'].value_counts()
    total = len(obs)
    print(f"\n{model_name} summary:")
    for ct, cnt in counts.items():
        pct = cnt / total * 100
        print(f"  {ct}: {cnt} ({pct:.2f}%)")

# ----------------------------------------------------------------------
# 5. Main execution
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # ---------- UPDATE THESE PATHS ----------
    # SD model
    file_path_sd = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/sd_model_5/new_segmentation_19_5/analysis_outputs/filtered/annotated_25-08-25_15-18_sd_19_model_5_filtered_cell.hdf5"
    annot_path_sd = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/sd_model_5/new_segmentation_19_5/analysis_outputs/filtered/annotations_19.csv"

    # Chenxi model
    file_path_chenxi = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/chenxi_model_1/16-09-25_16-05_chenxi_cellbound3v1_cell_annotated.hdf5"
    annot_path_chenxi = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/chenxi_model_1/annotations_19_chenxi.csv"

    # Process both models
    print("Processing SD model...")
    data_sd, df_sd = process_model(file_path_sd, annot_path_sd, "SD Model")

    print("\nProcessing Chenxi model...")
    data_chenxi, df_chenxi = process_model(file_path_chenxi, annot_path_chenxi, "Chenxi Model")

    # Combine for volume violin plot
    combined_df = pd.concat([df_sd, df_chenxi], ignore_index=True)
    combined_df = combined_df.dropna(subset=['volume'])

    print(f"\nChenxi model cells: {len(df_chenxi)}")
    print(f"SD model cells: {len(df_sd)}")
    print(f"Combined cells (after dropping NaN volumes): {len(combined_df)}")

    # ---------- 1. Violin plot: volume by cell type and model ----------
    plt.figure(figsize=(12, 8))
    sns.violinplot(data=combined_df, x='cell_type', y='volume', hue='model',
                   palette={'Chenxi Model': 'blue', 'SD Model': 'orange'},
                   split=True, inner="quartile")
    plt.title('Cell Volume Distributions by Cell Type and Model (Violin Plot)')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # ---------- 2. Spatial plots for each model ----------
    plot_spatial_sox3(data_sd, "SD Model")
    plot_spatial_sox17a(data_sd, "SD Model")
    plot_spatial_cell_types(data_sd, "SD Model")

    plot_spatial_sox3(data_chenxi, "Chenxi Model")
    plot_spatial_sox17a(data_chenxi, "Chenxi Model")
    plot_spatial_cell_types(data_chenxi, "Chenxi Model")

    # ---------- 3. Cumulative distribution plot (volume range 1000–30000) ----------
    plot_cumulative_volumes(obs_sd=data_sd['obs'],
                            obs_chenxi=data_chenxi['obs'],
                            lower=1000, upper=30000,
                            label_sd='SD', label_chenxi='Chenxi')

       # ---- 4. SOX17 vs Volume plots (500–30000) ----
    plot_sox17_vs_volume(obs_sd=data_sd['obs'],
                         obs_chenxi=data_chenxi['obs'],
                         sox17_threshold=5,
                         vol_lower=500, vol_upper=30000,
                         label_sd='SD', label_chenxi='Chenxi')

    # ---------- 5. Print cell type summary tables ----------
    print_cell_type_summary(data_sd['obs'], "SD Model")
    print_cell_type_summary(data_chenxi['obs'], "Chenxi Model")    
    
    
    
    

















#########################################3

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_reduced_data.py
Load MERSCOPE HDF5 + annotations, filter to 'ctl' cells,
classify by SOX3.S and SOX17A.S, and save a reduced CSV
containing only the columns needed for downstream analysis.
"""

import h5py
import numpy as np
import pandas as pd

def load_merscope_hdf5(file_path):
    """Load MERSCOPE HDF5 structure."""
    with h5py.File(file_path, 'r') as f:
        X = np.array(f['X'][:])
        print(f"Expression matrix shape: {X.shape}")
        obs_data = {}
        for key in f['obs'].keys():
            if key not in ['__categories', '_index']:
                obs_data[key] = np.array(f['obs'][key])
        var_index = np.array(f['var']['_index'][:]).astype(str)
        if 'spatial' in f['obsm']:
            spatial_coords = np.array(f['obsm']['spatial'][:])
        elif 'center_x' in obs_data and 'center_y' in obs_data:
            spatial_coords = np.column_stack([obs_data['center_x'], obs_data['center_y']])
        else:
            spatial_coords = None
        obs_df = pd.DataFrame(obs_data)
        if '_index' in f['obs']:
            obs_df.index = np.array(f['obs']['_index'][:]).astype(str)
        return {
            'expression': X,
            'obs': obs_df,
            'var': var_index,
            'spatial': spatial_coords
        }

def process_and_save(file_path, annotations_path, model_name, output_csv):
    """Process one model and save reduced CSV."""
    data = load_merscope_hdf5(file_path)
    annotations = pd.read_csv(annotations_path)

    obs = data['obs']
    obs['EntityID'] = obs.index

    # Merge annotations
    obs = obs.merge(
        annotations[['Custom cell groups', 'stage', 'condition']],
        on='Custom cell groups',
        how='left'
    )

    # Remove rows with missing stage/condition
    valid_mask = obs['stage'].notna() & obs['condition'].notna()
    obs = obs[valid_mask]
    expr = data['expression'][valid_mask]
    spatial = data['spatial'][valid_mask]

    # Keep only 'ctl' condition
    ctl_mask = obs['condition'] == 'ctl'
    obs = obs[ctl_mask]
    expr = expr[ctl_mask]
    spatial = spatial[ctl_mask]

    # Extract SOX3 and SOX17A expression
    var_names = data['var']
    sox3_idx = np.where(var_names == 'SOX3.S')[0][0]
    sox17a_idx = np.where(var_names == 'SOX17A.S')[0][0]
    sox3_exp = expr[:, sox3_idx]
    sox17a_exp = expr[:, sox17a_idx]

    sox3_th = 9
    sox17a_th = 5
    sox3_pos = sox3_exp >= sox3_th
    sox17a_pos = sox17a_exp >= sox17a_th
    double_pos = sox3_pos & sox17a_pos

    # Classify
    cell_types = []
    for i in range(len(sox3_exp)):
        if double_pos[i]:
            ct = 'Double Positive'
        elif sox3_pos[i]:
            ct = 'SOX3+ (Ectoderm)'
        elif sox17a_pos[i]:
            ct = 'SOX17A+ (Endoderm)'
        else:
            ct = 'Negative'
        cell_types.append(ct)

    # Build reduced DataFrame
    reduced_df = pd.DataFrame({
        'EntityID': obs.index,
        'volume': obs['volume'].values if 'volume' in obs.columns else np.nan,
        'x': spatial[:, 0],
        'y': spatial[:, 1],
        'sox3_expression': sox3_exp,
        'sox17a_expression': sox17a_exp,
        'cell_type': cell_types,
        'model': model_name
    })

    # Drop rows with missing volume
    reduced_df = reduced_df.dropna(subset=['volume'])

    # Save to CSV
    reduced_df.to_csv(output_csv, index=False)
    print(f"Saved {len(reduced_df)} cells for {model_name} to {output_csv}")

if __name__ == "__main__":
    # ---------- SET YOUR PATHS HERE ----------
    # SD model
    file_path_sd = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/sd_model_5/new_segmentation_19_5/analysis_outputs/filtered/annotated_25-08-25_15-18_sd_19_model_5_filtered_cell.hdf5"
    annot_path_sd = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/sd_model_5/new_segmentation_19_5/analysis_outputs/filtered/annotations_19.csv"
    # Chenxi model
    file_path_chenxi = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/chenxi_model_1/16-09-25_16-05_chenxi_cellbound3v1_cell_annotated.hdf5"
    annot_path_chenxi = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/chenxi_model_1/annotations_19_chenxi.csv"

    # Output CSVs (choose any names)
    output_sd = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/Technical_paper/Fig_source_codes/Fig4C_S4B_S4C_S4D/reduced_sd_model.csv"
    output_chenxi = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/Technical_paper/Fig_source_codes/Fig4C_S4B_S4C_S4D/reduced_chenxi_model.csv"

    process_and_save(file_path_sd, annot_path_sd, "SD Model", output_sd)
    process_and_save(file_path_chenxi, annot_path_chenxi, "Chenxi Model", output_chenxi)





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_reduced_data.py
Load MERSCOPE HDF5 + annotations, filter to 'ctl' cells,
extract SOX3 and SOX17A expression, and save a reduced CSV
with raw expression counts (no cell types).
"""

import h5py
import numpy as np
import pandas as pd

def load_merscope_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        X = np.array(f['X'][:])
        print(f"Expression matrix shape: {X.shape}")
        obs_data = {}
        for key in f['obs'].keys():
            if key not in ['__categories', '_index']:
                obs_data[key] = np.array(f['obs'][key])
        var_index = np.array(f['var']['_index'][:]).astype(str)
        if 'spatial' in f['obsm']:
            spatial_coords = np.array(f['obsm']['spatial'][:])
        elif 'center_x' in obs_data and 'center_y' in obs_data:
            spatial_coords = np.column_stack([obs_data['center_x'], obs_data['center_y']])
        else:
            spatial_coords = None
        obs_df = pd.DataFrame(obs_data)
        if '_index' in f['obs']:
            obs_df.index = np.array(f['obs']['_index'][:]).astype(str)
        return {
            'expression': X,
            'obs': obs_df,
            'var': var_index,
            'spatial': spatial_coords
        }

def process_and_save(file_path, annotations_path, model_name, output_csv):
    data = load_merscope_hdf5(file_path)
    annotations = pd.read_csv(annotations_path)

    obs = data['obs']
    obs['EntityID'] = obs.index

    obs = obs.merge(
        annotations[['Custom cell groups', 'stage', 'condition']],
        on='Custom cell groups',
        how='left'
    )

    # Remove missing stage/condition
    valid_mask = obs['stage'].notna() & obs['condition'].notna()
    obs = obs[valid_mask]
    expr = data['expression'][valid_mask]
    spatial = data['spatial'][valid_mask]

    # Keep only 'ctl'
    ctl_mask = obs['condition'] == 'ctl'
    obs = obs[ctl_mask]
    expr = expr[ctl_mask]
    spatial = spatial[ctl_mask]

    # Gene indices
    var_names = data['var']
    sox3_idx = np.where(var_names == 'SOX3.S')[0][0]
    sox17a_idx = np.where(var_names == 'SOX17A.S')[0][0]
    sox3_exp = expr[:, sox3_idx]
    sox17a_exp = expr[:, sox17a_idx]

    # Build reduced DataFrame (raw expression, no cell types)
    reduced_df = pd.DataFrame({
        'EntityID': obs.index,
        'volume': obs['volume'].values if 'volume' in obs.columns else np.nan,
        'x': spatial[:, 0],
        'y': spatial[:, 1],
        'sox3_expression': sox3_exp,
        'sox17a_expression': sox17a_exp,
        'model': model_name
    })

    reduced_df = reduced_df.dropna(subset=['volume'])
    reduced_df.to_csv(output_csv, index=False)
    print(f"Saved {len(reduced_df)} cells for {model_name} to {output_csv}")

if __name__ == "__main__":
    # ---------- SET YOUR PATHS ----------
    file_path_sd = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/sd_model_5/new_segmentation_19_5/analysis_outputs/filtered/annotated_25-08-25_15-18_sd_19_model_5_filtered_cell.hdf5"
    annot_path_sd = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/sd_model_5/new_segmentation_19_5/analysis_outputs/filtered/annotations_19.csv"
    file_path_chenxi = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/chenxi_model_1/16-09-25_16-05_chenxi_cellbound3v1_cell_annotated.hdf5"
    annot_path_chenxi = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/chenxi_model_1/annotations_19_chenxi.csv"

    

# Output CSVs (choose any names)
    output_sd = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/Technical_paper/Fig_source_codes/Fig4C_S4B_S4C_S4D/reduced_sd_model.csv"
    output_chenxi = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/Technical_paper/Fig_source_codes/Fig4C_S4B_S4C_S4D/reduced_chenxi_model.csv"

    process_and_save(file_path_sd, annot_path_sd, "SD Model", output_sd)
    process_and_save(file_path_chenxi, annot_path_chenxi, "Chenxi Model", output_chenxi)
    
    
    
    
    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_reduced_data.py
Load MERSCOPE HDF5 + annotations, filter to 'ctl' cells,
extract SOX3 and SOX17A expression, and save a reduced CSV
with raw expression counts (no cell types).
Model names: Chenxi → Model_1, SD → Model_2
"""

import h5py
import numpy as np
import pandas as pd

def load_merscope_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        X = np.array(f['X'][:])
        print(f"Expression matrix shape: {X.shape}")
        obs_data = {}
        for key in f['obs'].keys():
            if key not in ['__categories', '_index']:
                obs_data[key] = np.array(f['obs'][key])
        var_index = np.array(f['var']['_index'][:]).astype(str)
        if 'spatial' in f['obsm']:
            spatial_coords = np.array(f['obsm']['spatial'][:])
        elif 'center_x' in obs_data and 'center_y' in obs_data:
            spatial_coords = np.column_stack([obs_data['center_x'], obs_data['center_y']])
        else:
            spatial_coords = None
        obs_df = pd.DataFrame(obs_data)
        if '_index' in f['obs']:
            obs_df.index = np.array(f['obs']['_index'][:]).astype(str)
        return {
            'expression': X,
            'obs': obs_df,
            'var': var_index,
            'spatial': spatial_coords
        }

def process_and_save(file_path, annotations_path, model_name, output_csv):
    data = load_merscope_hdf5(file_path)
    annotations = pd.read_csv(annotations_path)

    obs = data['obs']
    obs['EntityID'] = obs.index

    obs = obs.merge(
        annotations[['Custom cell groups', 'stage', 'condition']],
        on='Custom cell groups',
        how='left'
    )

    # Remove missing stage/condition
    valid_mask = obs['stage'].notna() & obs['condition'].notna()
    obs = obs[valid_mask]
    expr = data['expression'][valid_mask]
    spatial = data['spatial'][valid_mask]

    # Keep only 'ctl'
    ctl_mask = obs['condition'] == 'ctl'
    obs = obs[ctl_mask]
    expr = expr[ctl_mask]
    spatial = spatial[ctl_mask]

    # Gene indices
    var_names = data['var']
    sox3_idx = np.where(var_names == 'SOX3.S')[0][0]
    sox17a_idx = np.where(var_names == 'SOX17A.S')[0][0]
    sox3_exp = expr[:, sox3_idx]
    sox17a_exp = expr[:, sox17a_idx]

    # Build reduced DataFrame (raw expression, no cell types)
    reduced_df = pd.DataFrame({
        'EntityID': obs.index,
        'volume': obs['volume'].values if 'volume' in obs.columns else np.nan,
        'x': spatial[:, 0],
        'y': spatial[:, 1],
        'sox3_expression': sox3_exp,
        'sox17a_expression': sox17a_exp,
        'model': model_name   # "Model_1" or "Model_2"
    })

    reduced_df = reduced_df.dropna(subset=['volume'])
    reduced_df.to_csv(output_csv, index=False)
    print(f"Saved {len(reduced_df)} cells for {model_name} to {output_csv}")

if __name__ == "__main__":
    # ---------- SET YOUR PATHS ----------
    file_path_sd = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/sd_model_5/new_segmentation_19_5/analysis_outputs/filtered/annotated_25-08-25_15-18_sd_19_model_5_filtered_cell.hdf5"
    annot_path_sd = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/sd_model_5/new_segmentation_19_5/analysis_outputs/filtered/annotations_19.csv"
    file_path_chenxi = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/chenxi_model_1/16-09-25_16-05_chenxi_cellbound3v1_cell_annotated.hdf5"
    annot_path_chenxi = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/chenxi_model_1/annotations_19_chenxi.csv"

    # Output CSVs (choose any names)
    output_sd = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/Technical_paper/Fig_source_codes/Fig4C_S4B_S4C_S4D/reduced_Model_2.csv"
    output_chenxi = "/Users/sdas/Desktop/Data/MERSCOPE/Resegmentation/19/Technical_paper/Fig_source_codes/Fig4C_S4B_S4C_S4D/reduced_Model_1.csv"

    # Now passing "Model_2" and "Model_1" as the model names
    process_and_save(file_path_sd, annot_path_sd, "Model_2", output_sd)
    process_and_save(file_path_chenxi, annot_path_chenxi, "Model_1", output_chenxi)

    