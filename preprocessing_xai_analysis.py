"""
Preprocessing × XAI Comprehensive Analysis
============================================
Analyzes how each preprocessing factor affects XAI metrics (not just accuracy).
Includes:
  1. Per-factor XAI impact (AC, SC, PC, Entropy) with effect sizes
  2. Composite preprocessing pipeline effects
  3. Resolution effect statistical tests (Kruskal-Wallis + pairwise)
  4. Channel attention effect statistical tests
  5. Visualizations

Uses saved Grad-CAM++ data and existing metric results.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('outputs/analysis')

# ── Dataset mapping ─────────────────────────────────────────────────
# d1=original, d2=normalized, d3=resized(stretch), d4=resized(preserve),
# d5=normalized+resized(stretch), d6=normalized+resized(preserve),
# d7=deblurred, d8=deblurred+normalized, d9=deblurred+resized(stretch),
# d10=deblurred+resized(preserve), d11=deblurred+normalized+resized(stretch),
# d12=deblurred+normalized+resized(preserve)

DS_LABELS = {
    'd1': 'Original',
    'd2': 'Norm',
    'd3': 'Resize(S)',
    'd4': 'Resize(P)',
    'd5': 'Norm+Resize(S)',
    'd6': 'Norm+Resize(P)',
    'd7': 'Deblur',
    'd8': 'Deblur+Norm',
    'd9': 'Deblur+Resize(S)',
    'd10': 'Deblur+Resize(P)',
    'd11': 'Deblur+Norm+Resize(S)',
    'd12': 'Deblur+Norm+Resize(P)',
}

# Preprocessing factor pairs (differing by exactly one factor)
FACTOR_PAIRS = {
    'Deblurring': [
        ('d1', 'd7'), ('d2', 'd8'), ('d3', 'd9'),
        ('d4', 'd10'), ('d5', 'd11'), ('d6', 'd12'),
    ],
    'Normalization': [
        ('d1', 'd2'), ('d3', 'd5'), ('d4', 'd6'), ('d7', 'd8'),
    ],
    'Resizing': [
        ('d1', 'd3'), ('d2', 'd5'), ('d7', 'd9'), ('d8', 'd11'),
    ],
    'Padding': [
        ('d3', 'd4'), ('d5', 'd6'), ('d9', 'd10'), ('d11', 'd12'),
    ],
}

# Composite pipeline: minimal → maximal preprocessing
COMPOSITE_PIPELINE = [
    ('d1', 'Original'),
    ('d7', '+Deblur'),
    ('d3', '+Resize'),
    ('d2', '+Norm'),
    ('d9', '+Deblur+Resize'),
    ('d12', '+Deblur+Norm+Resize(P)'),
]

XAI_METRICS = ['AC@30', 'Entropy', 'PC@10', 'SC@30']

# Models (17)
MODELS_17 = [
    'shufflenet_v2_x1_0', 'shufflenet_v2_x0_5',
    'mobilenet_v3_small', 'efficientnet_b0', 'regnet_y_400mf',
    'shufflenet_v2_x1_0_hr', 'shufflenet_v2_x0_5_hr',
    'mobilenet_v3_small_hr', 'efficientnet_b0_hr',
    'shufflenet_v2_x1_0_28', 'shufflenet_v2_x0_5_28',
    'mobilenet_v3_small_28', 'efficientnet_b0_28',
    'varroanet', 'varroanet_hr', 'varroanet_28', 'varroanet_56',
]

DISPLAY_NAMES = {
    'shufflenet_v2_x1_0': 'ShuffleNet-1.0', 'shufflenet_v2_x0_5': 'ShuffleNet-0.5',
    'mobilenet_v3_small': 'MobileNetV3-S', 'efficientnet_b0': 'EfficientNet-B0',
    'regnet_y_400mf': 'RegNet-Y-400MF',
    'shufflenet_v2_x1_0_hr': 'ShuffleNet-1.0-HR', 'shufflenet_v2_x0_5_hr': 'ShuffleNet-0.5-HR',
    'mobilenet_v3_small_hr': 'MobileNetV3-S-HR', 'efficientnet_b0_hr': 'EfficientNet-B0-HR',
    'shufflenet_v2_x1_0_28': 'ShuffleNet-1.0-28', 'shufflenet_v2_x0_5_28': 'ShuffleNet-0.5-28',
    'mobilenet_v3_small_28': 'MobileNetV3-S-28', 'efficientnet_b0_28': 'EfficientNet-B0-28',
    'varroanet': 'VarroaNet', 'varroanet_hr': 'VarroaNet-HR',
    'varroanet_28': 'VarroaNet-28', 'varroanet_56': 'VarroaNet-56',
}

RES_MAP = {}
for m in MODELS_17:
    if '56' in m:
        RES_MAP[m] = '56×56'
    elif '28' in m:
        RES_MAP[m] = '28×28'
    elif 'hr' in m:
        RES_MAP[m] = '14×14'
    else:
        RES_MAP[m] = '7×7'


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 9,
        'axes.linewidth': 0.8, 'axes.grid': True,
        'grid.alpha': 0.3, 'grid.linewidth': 0.5, 'figure.dpi': 300,
    })


def load_data():
    """Load the raw threshold sensitivity data (17 models)."""
    raw_path = OUTPUT_DIR / 'threshold_sensitivity_raw_17models.csv'
    if raw_path.exists():
        df = pd.read_csv(raw_path)
        print(f"Loaded {len(df)} rows from cached data")
        return df
    else:
        print("ERROR: Run threshold_sensitivity_viz.py first to generate raw data.")
        return None


def cohens_d(x, y):
    """Compute Cohen's d for paired samples."""
    diff = x - y
    return diff.mean() / diff.std() if diff.std() > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
# 1. Per-Factor XAI Impact Analysis
# ═══════════════════════════════════════════════════════════════════
def per_factor_xai_analysis(df):
    """For each preprocessing factor, compute effect on each XAI metric."""
    print("\n" + "=" * 70)
    print("1. PER-FACTOR XAI IMPACT ANALYSIS")
    print("=" * 70)

    metric_cols = {
        'AC@30': 'AC@30', 'Entropy': 'entropy',
        'PC@10': 'PC@10', 'SC@30': 'SC@30',
    }

    results = []

    for factor_name, pairs in FACTOR_PAIRS.items():
        for metric_label, metric_col in metric_cols.items():
            deltas_all = []
            for ds_base, ds_treated in pairs:
                base_vals = df[df.dataset == ds_base].groupby('model')[metric_col].mean()
                treat_vals = df[df.dataset == ds_treated].groupby('model')[metric_col].mean()
                common = base_vals.index.intersection(treat_vals.index)
                if len(common) < 3:
                    continue
                delta = treat_vals[common] - base_vals[common]
                deltas_all.extend(delta.values)

            if len(deltas_all) < 3:
                continue

            deltas_arr = np.array(deltas_all)
            mean_delta = deltas_arr.mean()
            std_delta = deltas_arr.std()
            d = mean_delta / std_delta if std_delta > 0 else 0.0

            # Wilcoxon on deltas (test if different from 0)
            if len(deltas_arr) >= 5:
                try:
                    stat, p = stats.wilcoxon(deltas_arr)
                except:
                    p = 1.0
            else:
                p = 1.0

            # Effect size label
            ad = abs(d)
            if ad < 0.2:
                eff = 'Negligible'
            elif ad < 0.5:
                eff = 'Small'
            elif ad < 0.8:
                eff = 'Medium'
            else:
                eff = 'Large'

            results.append({
                'Factor': factor_name,
                'Metric': metric_label,
                'Mean Delta': mean_delta,
                'Std Delta': std_delta,
                "Cohen's d": d,
                'Effect Size': eff,
                'p-value': p,
                'N': len(deltas_arr),
            })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    results_df.to_csv(OUTPUT_DIR / 'preprocessing_xai_per_factor.csv', index=False)
    print(f"\nSaved: preprocessing_xai_per_factor.csv")

    # Visualization: heatmap of Cohen's d
    setup_style()
    pivot = results_df.pivot(index='Factor', columns='Metric', values="Cohen's d")
    factor_order = ['Resizing', 'Deblurring', 'Normalization', 'Padding']
    metric_order = ['AC@30', 'Entropy', 'PC@10', 'SC@30']
    pivot = pivot.reindex(index=factor_order, columns=metric_order)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-0.5, vmax=0.5)
    ax.set_xticks(range(len(metric_order)))
    ax.set_xticklabels(metric_order, fontsize=9)
    ax.set_yticks(range(len(factor_order)))
    ax.set_yticklabels(factor_order, fontsize=9)
    for i in range(len(factor_order)):
        for j in range(len(metric_order)):
            val = pivot.values[i, j]
            ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(val) > 0.3 else 'black')
    ax.set_title("Cohen's d: Preprocessing Factor Effect on XAI Metrics")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_preprocessing_xai_cohens_d.png', bbox_inches='tight')
    plt.close()
    print("  fig_preprocessing_xai_cohens_d.png")

    return results_df


# ═══════════════════════════════════════════════════════════════════
# 2. Composite Preprocessing Pipeline Effects
# ═══════════════════════════════════════════════════════════════════
def composite_pipeline_analysis(df):
    """Show how XAI metrics change through the preprocessing pipeline."""
    print("\n" + "=" * 70)
    print("2. COMPOSITE PREPROCESSING PIPELINE EFFECTS")
    print("=" * 70)

    metric_cols = {'AC@30': 'AC@30', 'Entropy': 'entropy', 'PC@10': 'PC@10', 'SC@30': 'SC@30'}
    rows = []
    for ds_id, label in COMPOSITE_PIPELINE:
        subset = df[df.dataset == ds_id]
        if len(subset) == 0:
            continue
        row = {'Dataset': ds_id, 'Pipeline': label}
        for ml, mc in metric_cols.items():
            row[ml] = subset[mc].mean()
        rows.append(row)

    pipeline_df = pd.DataFrame(rows)
    print(pipeline_df.to_string(index=False))
    pipeline_df.to_csv(OUTPUT_DIR / 'preprocessing_xai_composite_pipeline.csv', index=False)

    # Also: full composite vs. original by resolution group
    print("\n--- Composite vs. Original by Resolution ---")
    for res in ['7×7', '14×14', '28×28', '56×56']:
        sub_orig = df[(df.dataset == 'd1') & (df.resolution == res)]
        sub_comp = df[(df.dataset == 'd12') & (df.resolution == res)]
        if len(sub_orig) == 0 or len(sub_comp) == 0:
            continue
        print(f"\n  {res}:")
        for ml, mc in metric_cols.items():
            v_orig = sub_orig[mc].mean()
            v_comp = sub_comp[mc].mean()
            delta = v_comp - v_orig
            pct = (delta / v_orig * 100) if v_orig != 0 else 0
            print(f"    {ml}: {v_orig:.4f} → {v_comp:.4f} (Δ={delta:+.4f}, {pct:+.1f}%)")

    # Visualization: grouped bar chart
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for idx, (ml, mc) in enumerate(metric_cols.items()):
        ax = axes[idx // 2][idx % 2]
        vals = pipeline_df[ml].values
        labels = pipeline_df['Pipeline'].values
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(vals)))
        bars = ax.bar(range(len(vals)), vals, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=7)
        ax.set_ylabel(ml)
        ax.set_title(f'{ml} across Preprocessing Pipeline')
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{v:.4f}', ha='center', va='bottom', fontsize=6.5)
    fig.suptitle('XAI Metrics: Preprocessing Pipeline Progression', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_preprocessing_xai_pipeline.png', bbox_inches='tight')
    plt.close()
    print("\n  fig_preprocessing_xai_pipeline.png")

    return pipeline_df


# ═══════════════════════════════════════════════════════════════════
# 3. Resolution Effect Statistical Tests
# ═══════════════════════════════════════════════════════════════════
def resolution_statistical_tests(df):
    """Kruskal-Wallis + pairwise Mann-Whitney U for resolution effects on XAI."""
    print("\n" + "=" * 70)
    print("3. RESOLUTION EFFECT STATISTICAL TESTS")
    print("=" * 70)

    metric_cols = {'AC@30': 'AC@30', 'Entropy': 'entropy', 'PC@10': 'PC@10', 'SC@30': 'SC@30'}
    res_levels = ['7×7', '14×14', '28×28', '56×56']

    results = []
    for ml, mc in metric_cols.items():
        groups = [df[df.resolution == r][mc].values for r in res_levels if len(df[df.resolution == r]) > 0]
        available_res = [r for r in res_levels if len(df[df.resolution == r]) > 0]

        # Kruskal-Wallis
        if len(groups) >= 3:
            h_stat, kw_p = stats.kruskal(*groups)
            print(f"\n{ml}: Kruskal-Wallis H={h_stat:.2f}, p={kw_p:.6f}")
            results.append({
                'Metric': ml, 'Test': 'Kruskal-Wallis',
                'Groups': ' vs '.join(available_res),
                'Statistic': h_stat, 'p-value': kw_p,
            })

            # Pairwise Mann-Whitney U
            for i in range(len(available_res)):
                for j in range(i + 1, len(available_res)):
                    u_stat, mw_p = stats.mannwhitneyu(groups[i], groups[j], alternative='two-sided')
                    d = (groups[i].mean() - groups[j].mean()) / np.sqrt(
                        (groups[i].std()**2 + groups[j].std()**2) / 2) if (groups[i].std() + groups[j].std()) > 0 else 0
                    sig = '***' if mw_p < 0.001 else '**' if mw_p < 0.01 else '*' if mw_p < 0.05 else 'ns'
                    print(f"  {available_res[i]} vs {available_res[j]}: "
                          f"U={u_stat:.0f}, p={mw_p:.6f} {sig}, Cohen's d={d:.3f}")
                    results.append({
                        'Metric': ml, 'Test': 'Mann-Whitney U',
                        'Groups': f'{available_res[i]} vs {available_res[j]}',
                        'Statistic': u_stat, 'p-value': mw_p, "Cohen's d": d,
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'resolution_xai_statistical_tests.csv', index=False)
    print(f"\nSaved: resolution_xai_statistical_tests.csv")

    # Visualization: box plot per metric per resolution
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    res_colors = {'7×7': '#d62728', '14×14': '#ff7f0e', '28×28': '#2ca02c', '56×56': '#1f77b4'}

    for idx, (ml, mc) in enumerate(metric_cols.items()):
        ax = axes[idx // 2][idx % 2]
        data_for_box = []
        labels_for_box = []
        colors_for_box = []
        for r in res_levels:
            vals = df[df.resolution == r][mc].values
            if len(vals) > 0:
                data_for_box.append(vals)
                labels_for_box.append(r)
                colors_for_box.append(res_colors[r])

        bp = ax.boxplot(data_for_box, labels=labels_for_box, patch_artist=True,
                        showfliers=False, widths=0.6)
        for patch, color in zip(bp['boxes'], colors_for_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(ml)
        ax.set_title(f'{ml} by Feature Map Resolution')

    fig.suptitle('XAI Metrics Distribution by Resolution', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_resolution_xai_boxplot.png', bbox_inches='tight')
    plt.close()
    print("  fig_resolution_xai_boxplot.png")

    return results_df


# ═══════════════════════════════════════════════════════════════════
# 4. Channel Attention Effect Statistical Tests
# ═══════════════════════════════════════════════════════════════════
def channel_attention_tests(df):
    """Compare VarroaNet vs ShuffleNet-V2-x1.0 at each resolution."""
    print("\n" + "=" * 70)
    print("4. CHANNEL ATTENTION EFFECT (VarroaNet vs ShuffleNet-1.0)")
    print("=" * 70)

    pairs = [
        ('varroanet', 'shufflenet_v2_x1_0', '7×7'),
        ('varroanet_hr', 'shufflenet_v2_x1_0_hr', '14×14'),
        ('varroanet_28', 'shufflenet_v2_x1_0_28', '28×28'),
    ]

    metric_cols = {'AC@30': 'AC@30', 'Entropy': 'entropy', 'PC@10': 'PC@10', 'SC@30': 'SC@30'}
    results = []

    for vn_model, sn_model, res in pairs:
        print(f"\n--- {res}: {DISPLAY_NAMES[vn_model]} vs {DISPLAY_NAMES[sn_model]} ---")
        for ml, mc in metric_cols.items():
            vn_vals = df[df.model == vn_model][mc].values
            sn_vals = df[df.model == sn_model][mc].values

            if len(vn_vals) < 10 or len(sn_vals) < 10:
                continue

            u_stat, p = stats.mannwhitneyu(vn_vals, sn_vals, alternative='two-sided')
            d = (vn_vals.mean() - sn_vals.mean()) / np.sqrt(
                (vn_vals.std()**2 + sn_vals.std()**2) / 2) if (vn_vals.std() + sn_vals.std()) > 0 else 0
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'

            print(f"  {ml}: VN={vn_vals.mean():.4f}, SN={sn_vals.mean():.4f}, "
                  f"Δ={vn_vals.mean()-sn_vals.mean():.4f}, d={d:.3f}, p={p:.6f} {sig}")

            results.append({
                'Resolution': res,
                'Metric': ml,
                'VarroaNet_mean': vn_vals.mean(),
                'ShuffleNet_mean': sn_vals.mean(),
                'Delta': vn_vals.mean() - sn_vals.mean(),
                "Cohen's d": d,
                'p-value': p,
                'Significant': sig,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_DIR / 'channel_attention_xai_tests.csv', index=False)
    print(f"\nSaved: channel_attention_xai_tests.csv")

    # Visualization: paired bar chart
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    res_list = ['7×7', '14×14', '28×28']

    for idx, (ml, mc) in enumerate(metric_cols.items()):
        ax = axes[idx // 2][idx % 2]
        sub = results_df[results_df.Metric == ml]
        x = np.arange(len(res_list))
        w = 0.35
        vn_vals = [sub[sub.Resolution == r]['VarroaNet_mean'].values[0] if r in sub.Resolution.values else 0 for r in res_list]
        sn_vals = [sub[sub.Resolution == r]['ShuffleNet_mean'].values[0] if r in sub.Resolution.values else 0 for r in res_list]

        ax.bar(x - w/2, vn_vals, w, label='VarroaNet', color='#2ca02c', edgecolor='black', linewidth=0.5)
        ax.bar(x + w/2, sn_vals, w, label='ShuffleNet-1.0', color='#d62728', edgecolor='black', linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(res_list)
        ax.set_ylabel(ml)
        ax.set_title(f'{ml}: Channel Attention Effect')
        ax.legend(bbox_to_anchor=(1.0, 1.0), loc='upper right', fontsize=7, frameon=True)

    fig.suptitle('VarroaNet (with CA) vs ShuffleNet-V2-x1.0 (without CA)', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_channel_attention_xai.png', bbox_inches='tight')
    plt.close()
    print("  fig_channel_attention_xai.png")

    return results_df


# ═══════════════════════════════════════════════════════════════════
# 5. VarroaNet Resolution Curve (4-point)
# ═══════════════════════════════════════════════════════════════════
def varroanet_resolution_curve(df):
    """Plot VarroaNet family XAI metrics across 4 resolutions."""
    print("\n" + "=" * 70)
    print("5. VARROANET 4-POINT RESOLUTION CURVE")
    print("=" * 70)

    vn_models = ['varroanet', 'varroanet_hr', 'varroanet_28', 'varroanet_56']
    res_labels = ['7×7', '14×14', '28×28', '56×56']
    metric_cols = {'AC@30': 'AC@30', 'Entropy': 'entropy', 'PC@10': 'PC@10', 'SC@30': 'SC@30'}

    rows = []
    for m, r in zip(vn_models, res_labels):
        sub = df[df.model == m]
        if len(sub) == 0:
            continue
        row = {'Model': DISPLAY_NAMES[m], 'Resolution': r}
        for ml, mc in metric_cols.items():
            row[ml] = sub[mc].mean()
        rows.append(row)
        print(f"  {DISPLAY_NAMES[m]} ({r}): "
              f"AC@30={sub['AC@30'].mean():.4f}, Ent={sub['entropy'].mean():.4f}, "
              f"PC@10={sub['PC@10'].mean():.4f}, SC@30={sub['SC@30'].mean():.4f}")

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(OUTPUT_DIR / 'varroanet_resolution_curve_xai.csv', index=False)

    # Visualization
    setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    x = range(len(curve_df))

    for idx, (ml, mc) in enumerate(metric_cols.items()):
        ax = axes[idx // 2][idx % 2]
        vals = curve_df[ml].values
        ax.plot(x, vals, 'o-', color='#1f77b4', linewidth=2, markersize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(res_labels)
        ax.set_ylabel(ml)
        ax.set_title(f'VarroaNet: {ml} across Resolutions')
        for i, v in enumerate(vals):
            ax.annotate(f'{v:.4f}', (i, v), textcoords='offset points',
                        xytext=(0, 10), ha='center', fontsize=7.5)

    fig.suptitle('VarroaNet Family: Annotation-Free XAI Metrics (4-Point Resolution Curve)', fontsize=10)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig_varroanet_resolution_curve_xai.png', bbox_inches='tight')
    plt.close()
    print("  fig_varroanet_resolution_curve_xai.png")

    return curve_df


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    df = load_data()
    if df is None:
        exit(1)

    # Rename columns to match expected names
    if 'AC@30' not in df.columns and 'AC@30' in [c for c in df.columns]:
        pass  # already correct

    r1 = per_factor_xai_analysis(df)
    r2 = composite_pipeline_analysis(df)
    r3 = resolution_statistical_tests(df)
    r4 = channel_attention_tests(df)
    r5 = varroanet_resolution_curve(df)

    print("\n" + "=" * 70)
    print("ALL ANALYSES COMPLETE")
    print("=" * 70)
    print("\nGenerated files:")
    for f in sorted(OUTPUT_DIR.glob('fig_*.png')):
        print(f"  [FIG] {f.name}")
    for f in sorted(OUTPUT_DIR.glob('*xai*.csv')) | sorted(OUTPUT_DIR.glob('*resolution*.csv')) | sorted(OUTPUT_DIR.glob('*channel*.csv')):
        print(f"  [CSV] {f.name}")
