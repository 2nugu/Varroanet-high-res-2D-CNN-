"""
Threshold Sensitivity Visualization (17 models including VarroaNet-56)
=====================================================================
Produces clean academic figures with legends outside (top-right).
Saves numerical data as CSV.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import ndimage
from scipy.stats import kendalltau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────
BASE_DIR = Path('outputs/gradcam')
OUTPUT_DIR = Path('outputs/analysis')
OUTPUT_DIR.mkdir(exist_ok=True)

DATASETS = [f'd{i}' for i in range(1, 13)]
MODELS = [
    'shufflenet_v2_x1_0', 'shufflenet_v2_x0_5',
    'mobilenet_v3_small', 'efficientnet_b0', 'regnet_y_400mf',
    'shufflenet_v2_x1_0_hr', 'shufflenet_v2_x0_5_hr',
    'mobilenet_v3_small_hr', 'efficientnet_b0_hr',
    'shufflenet_v2_x1_0_28', 'shufflenet_v2_x0_5_28',
    'mobilenet_v3_small_28', 'efficientnet_b0_28',
    'varroanet', 'varroanet_hr', 'varroanet_28', 'varroanet_56',
]

# Display names for plots
DISPLAY_NAMES = {
    'shufflenet_v2_x1_0': 'ShuffleNet-1.0',
    'shufflenet_v2_x0_5': 'ShuffleNet-0.5',
    'mobilenet_v3_small': 'MobileNetV3-S',
    'efficientnet_b0': 'EfficientNet-B0',
    'regnet_y_400mf': 'RegNet-Y-400MF',
    'shufflenet_v2_x1_0_hr': 'ShuffleNet-1.0-HR',
    'shufflenet_v2_x0_5_hr': 'ShuffleNet-0.5-HR',
    'mobilenet_v3_small_hr': 'MobileNetV3-S-HR',
    'efficientnet_b0_hr': 'EfficientNet-B0-HR',
    'shufflenet_v2_x1_0_28': 'ShuffleNet-1.0-28',
    'shufflenet_v2_x0_5_28': 'ShuffleNet-0.5-28',
    'mobilenet_v3_small_28': 'MobileNetV3-S-28',
    'efficientnet_b0_28': 'EfficientNet-B0-28',
    'varroanet': 'VarroaNet',
    'varroanet_hr': 'VarroaNet-HR',
    'varroanet_28': 'VarroaNet-28',
    'varroanet_56': 'VarroaNet-56',
}

# Resolution groups
RES_MAP = {}
for m in MODELS:
    if '56' in m:
        RES_MAP[m] = '56×56'
    elif '28' in m:
        RES_MAP[m] = '28×28'
    elif 'hr' in m:
        RES_MAP[m] = '14×14'
    else:
        RES_MAP[m] = '7×7'

# Color scheme by resolution
RES_COLORS = {'7×7': '#d62728', '14×14': '#ff7f0e', '28×28': '#2ca02c', '56×56': '#1f77b4'}
# Marker by architecture family
ARCH_MARKERS = {
    'ShuffleNet-1.0': 'o', 'ShuffleNet-0.5': 's',
    'MobileNetV3-S': '^', 'EfficientNet-B0': 'D',
    'RegNet-Y-400MF': 'v',
    'VarroaNet': '*',
}

def get_marker(model_name):
    dn = DISPLAY_NAMES[model_name]
    for prefix, marker in ARCH_MARKERS.items():
        if dn.startswith(prefix) or dn == prefix:
            return marker
    return 'o'

def get_color(model_name):
    return RES_COLORS[RES_MAP[model_name]]

AC_THRESHOLDS = [round(t, 1) for t in np.arange(0.2, 0.91, 0.1)]
SC_THRESHOLDS = [round(t, 1) for t in np.arange(0.2, 0.91, 0.1)]
PC_TOPK = [round(k, 2) for k in np.arange(0.05, 0.51, 0.05)]


# ── Metric Functions ───────────────────────────────────────────────
def activation_coverage(cam, threshold):
    return float(np.mean(cam >= threshold))

def spatial_compactness(cam, threshold):
    binary = (cam >= threshold).astype(np.uint8)
    total = binary.sum()
    if total == 0:
        return 0.0
    labeled, n = ndimage.label(binary)
    if n == 0:
        return 0.0
    sizes = ndimage.sum(binary, labeled, range(1, n + 1))
    return float(np.max(sizes) / total)

def peak_concentration(cam, top_k_fraction):
    total = cam.sum()
    if total == 0:
        return 0.0
    q = np.quantile(cam, 1.0 - top_k_fraction)
    return float(cam[cam >= q].sum() / total)

def heatmap_entropy(cam):
    flat = cam.flatten()
    if flat.sum() == 0:
        return 0.0
    hist, _ = np.histogram(flat, bins=50, range=(0, 1), density=True)
    hist = hist / hist.sum()
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


# ── Data Processing ────────────────────────────────────────────────
def process_all():
    records = []
    total_files = 0
    skipped = 0

    for ds in DATASETS:
        for model in MODELS:
            model_dir = BASE_DIR / ds / model
            if not model_dir.exists():
                continue
            npy_files = sorted(model_dir.glob('*_mite_cam_data.npy'))
            for npy_path in npy_files:
                cam = np.load(npy_path)
                if cam.max() == 0:
                    skipped += 1
                    continue
                if cam.max() > 1.0:
                    cam = cam / cam.max()
                total_files += 1

                row = {
                    'dataset': ds, 'model': model,
                    'resolution': RES_MAP[model],
                    'entropy': heatmap_entropy(cam),
                }
                for t in AC_THRESHOLDS:
                    row[f'AC@{int(t*100)}'] = activation_coverage(cam, t)
                for t in SC_THRESHOLDS:
                    row[f'SC@{int(t*100)}'] = spatial_compactness(cam, t)
                for k in PC_TOPK:
                    row[f'PC@{int(k*100)}'] = peak_concentration(cam, k)
                records.append(row)

    print(f"Processed {total_files} heatmaps, skipped {skipped} empty")
    return pd.DataFrame(records)


# ── Visualization ──────────────────────────────────────────────────
def setup_style():
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 9,
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.5,
        'figure.dpi': 300,
    })

def plot_ac_by_model(ac_summary):
    """Fig 1: AC across thresholds, one line per model."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    thresholds = [int(c.replace('AC@', '')) / 100 for c in ac_summary.columns]
    for model in ac_summary.index:
        vals = ac_summary.loc[model].values
        ax.plot(thresholds, vals,
                marker=get_marker(model), color=get_color(model),
                markersize=4, linewidth=1.2, label=DISPLAY_NAMES[model])

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Activation Coverage (AC)')
    ax.set_title('(a) Activation Coverage vs. Threshold by Model')
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=6.5,
              frameon=True, framealpha=0.9, ncol=1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_threshold_AC_by_model.png', bbox_inches='tight')
    plt.close()

def plot_sc_by_model(sc_summary):
    """Fig 2: SC across thresholds."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    thresholds = [int(c.replace('SC@', '')) / 100 for c in sc_summary.columns]
    for model in sc_summary.index:
        vals = sc_summary.loc[model].values
        ax.plot(thresholds, vals,
                marker=get_marker(model), color=get_color(model),
                markersize=4, linewidth=1.2, label=DISPLAY_NAMES[model])

    ax.set_xlabel('Threshold')
    ax.set_ylabel('Spatial Compactness (SC)')
    ax.set_title('(b) Spatial Compactness vs. Threshold by Model')
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=6.5,
              frameon=True, framealpha=0.9, ncol=1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_threshold_SC_by_model.png', bbox_inches='tight')
    plt.close()

def plot_pc_by_model(pc_summary):
    """Fig 3: PC across top-k%."""
    setup_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    topk = [int(c.replace('PC@', '')) for c in pc_summary.columns]
    for model in pc_summary.index:
        vals = pc_summary.loc[model].values
        ax.plot(topk, vals,
                marker=get_marker(model), color=get_color(model),
                markersize=4, linewidth=1.2, label=DISPLAY_NAMES[model])

    ax.set_xlabel('Top-k (%)')
    ax.set_ylabel('Peak Concentration (PC)')
    ax.set_title('(c) Peak Concentration vs. Top-k% by Model')
    ax.legend(bbox_to_anchor=(1.02, 1.0), loc='upper left', fontsize=6.5,
              frameon=True, framealpha=0.9, ncol=1)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_threshold_PC_by_model.png', bbox_inches='tight')
    plt.close()

def plot_resolution_level(res_ac, res_sc, res_pc):
    """Fig 4: Resolution-level averages (3 subplots)."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    res_order = ['7×7', '14×14', '28×28', '56×56']
    res_line_colors = {'7×7': '#d62728', '14×14': '#ff7f0e', '28×28': '#2ca02c', '56×56': '#1f77b4'}

    # AC
    ax = axes[0]
    thresholds = [int(c.replace('AC@', '')) / 100 for c in res_ac.columns]
    for res in res_order:
        if res in res_ac.index:
            ax.plot(thresholds, res_ac.loc[res].values, 'o-',
                    color=res_line_colors[res], linewidth=1.5, markersize=5, label=res)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('AC (mean)')
    ax.set_title('(a) AC by Resolution')
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=8, frameon=True)

    # SC
    ax = axes[1]
    thresholds = [int(c.replace('SC@', '')) / 100 for c in res_sc.columns]
    for res in res_order:
        if res in res_sc.index:
            ax.plot(thresholds, res_sc.loc[res].values, 's-',
                    color=res_line_colors[res], linewidth=1.5, markersize=5, label=res)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('SC (mean)')
    ax.set_title('(b) SC by Resolution')
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=8, frameon=True)

    # PC
    ax = axes[2]
    topk = [int(c.replace('PC@', '')) for c in res_pc.columns]
    for res in res_order:
        if res in res_pc.index:
            ax.plot(topk, res_pc.loc[res].values, '^-',
                    color=res_line_colors[res], linewidth=1.5, markersize=5, label=res)
    ax.set_xlabel('Top-k (%)')
    ax.set_ylabel('PC (mean)')
    ax.set_title('(c) PC by Resolution')
    ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_threshold_resolution_level.png', bbox_inches='tight')
    plt.close()

def plot_rank_stability(ac_ranks):
    """Fig 5: Rank heatmap across thresholds."""
    setup_style()
    display_index = [DISPLAY_NAMES[m] for m in ac_ranks.index]
    plot_data = ac_ranks.copy()
    plot_data.index = display_index

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(plot_data.values, cmap='RdYlGn_r', aspect='auto',
                   vmin=1, vmax=len(plot_data))

    ax.set_xticks(range(len(plot_data.columns)))
    ax.set_xticklabels(plot_data.columns, fontsize=8)
    ax.set_yticks(range(len(plot_data.index)))
    ax.set_yticklabels(plot_data.index, fontsize=7)

    for i in range(len(plot_data.index)):
        for j in range(len(plot_data.columns)):
            val = int(plot_data.values[i, j])
            color = 'white' if val <= 3 or val >= 15 else 'black'
            ax.text(j, i, str(val), ha='center', va='center', fontsize=6.5, color=color)

    ax.set_title('AC Model Rank Across Thresholds (1 = most focused)')
    ax.set_xlabel('Threshold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Rank')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_threshold_AC_rank_heatmap.png', bbox_inches='tight')
    plt.close()

def plot_kendall_tau(ac_ranks):
    """Fig 6: Kendall's tau bar chart."""
    setup_style()
    cols = ac_ranks.columns.tolist()
    ref = ac_ranks['AC@30']
    taus, pvals, labels = [], [], []
    for col in cols:
        if col == 'AC@30':
            continue
        tau, p = kendalltau(ref, ac_ranks[col])
        taus.append(tau)
        pvals.append(p)
        labels.append(col)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ['#2ca02c' if t >= 0.9 else '#ff7f0e' if t >= 0.7 else '#d62728' for t in taus]
    bars = ax.bar(labels, taus, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_ylabel("Kendall's τ")
    ax.set_xlabel('Threshold')
    ax.set_title("Kendall's τ Rank Correlation with AC@30")
    ax.set_ylim(0.5, 1.05)
    ax.axhline(y=0.9, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)

    for bar, tau, p in zip(bars, taus, pvals):
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{tau:.3f}\n{sig}', ha='center', va='bottom', fontsize=6.5)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_kendall_tau_AC.png', bbox_inches='tight')
    plt.close()


# ── Main ───────────────────────────────────────────────────────────
def create_summaries(df):
    ac_cols = [c for c in df.columns if c.startswith('AC@')]
    sc_cols = [c for c in df.columns if c.startswith('SC@')]
    pc_cols = [c for c in df.columns if c.startswith('PC@')]

    ac_summary = df.groupby('model')[ac_cols].mean()
    sc_summary = df.groupby('model')[sc_cols].mean()
    pc_summary = df.groupby('model')[pc_cols].mean()

    # Reorder
    model_order = [m for m in MODELS if m in ac_summary.index]
    ac_summary = ac_summary.loc[model_order]
    sc_summary = sc_summary.loc[model_order]
    pc_summary = pc_summary.loc[model_order]

    ac_ranks = ac_summary.rank(ascending=True)
    sc_ranks = sc_summary.rank(ascending=False)
    pc_ranks = pc_summary.rank(ascending=False)

    res_ac = df.groupby('resolution')[ac_cols].mean()
    res_sc = df.groupby('resolution')[sc_cols].mean()
    res_pc = df.groupby('resolution')[pc_cols].mean()

    return {
        'ac_summary': ac_summary, 'sc_summary': sc_summary, 'pc_summary': pc_summary,
        'ac_ranks': ac_ranks, 'sc_ranks': sc_ranks, 'pc_ranks': pc_ranks,
        'res_ac': res_ac, 'res_sc': res_sc, 'res_pc': res_pc,
    }


if __name__ == '__main__':
    print("Processing 17 models (incl. VarroaNet-56)...")
    df = process_all()

    print("Computing summaries...")
    tables = create_summaries(df)

    # Save CSVs
    for name, tbl in tables.items():
        tbl.to_csv(OUTPUT_DIR / f'threshold_sensitivity_{name}.csv')
    df.to_csv(OUTPUT_DIR / 'threshold_sensitivity_raw_17models.csv', index=False)
    print(f"Saved CSVs ({len(df)} rows)")

    # Generate figures
    print("Generating figures...")
    plot_ac_by_model(tables['ac_summary'])
    print("  fig_threshold_AC_by_model.png")
    plot_sc_by_model(tables['sc_summary'])
    print("  fig_threshold_SC_by_model.png")
    plot_pc_by_model(tables['pc_summary'])
    print("  fig_threshold_PC_by_model.png")
    plot_resolution_level(tables['res_ac'], tables['res_sc'], tables['res_pc'])
    print("  fig_threshold_resolution_level.png")
    plot_rank_stability(tables['ac_ranks'])
    print("  fig_threshold_AC_rank_heatmap.png")
    plot_kendall_tau(tables['ac_ranks'])
    print("  fig_kendall_tau_AC.png")

    # Print Kendall's tau summary
    ref = tables['ac_ranks']['AC@30']
    print("\n--- Kendall's τ (AC@30 ref, 17 models) ---")
    for col in tables['ac_ranks'].columns:
        if col == 'AC@30':
            continue
        tau, p = kendalltau(ref, tables['ac_ranks'][col])
        print(f"  AC@30 vs {col}: τ={tau:.3f}, p={p:.6f}")

    print("\nDone.")
