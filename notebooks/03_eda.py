"""
notebooks/03_eda.py
Exploratory Data Analysis
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import umap
import os

os.makedirs('figures', exist_ok=True)

print("Loading processed data...")
expr   = pd.read_csv('data/processed/expression_clean.csv',
                     index_col=0)
meth   = pd.read_csv('data/processed/methylation_pca.csv',
                     index_col=0)
cnv    = pd.read_csv('data/processed/cnv_clean.csv',
                     index_col=0)
labels = pd.read_csv('data/processed/labels.csv',
                     index_col=0).squeeze()

subtypes      = labels.values
subtype_names = sorted(set(subtypes))
colors = {
    'LumA':   '#378ADD',
    'LumB':   '#1D9E75',
    'Her2':   '#D85A30',
    'Basal':  '#E24B4A',
    'Normal': '#7F77DD'
}

print(f"Dataset: {len(labels)} patients")
print(f"Subtypes: {dict(pd.Series(subtypes).value_counts())}")

# ── Figure 1: 3-panel overview ───────────────────────
print("Creating overview figure...")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Subtype bar chart
counts = pd.Series(subtypes).value_counts()
c_list = [colors.get(s, 'gray') for s in counts.index]
bars   = axes[0].bar(counts.index, counts.values,
                     color=c_list, width=0.6,
                     edgecolor='white')
axes[0].set_title('PAM50 Subtype Distribution',
                  fontsize=13)
axes[0].set_ylabel('Number of patients')
axes[0].set_xlabel('Subtype')
for bar, v in zip(bars, counts.values):
    axes[0].text(
        bar.get_x()+bar.get_width()/2,
        v+3, str(v), ha='center', fontsize=11)

# Panel 2: UMAP of gene expression
print("Running UMAP on gene expression (~2 min)...")
scaler    = StandardScaler()
expr_sc   = scaler.fit_transform(expr)
reducer   = umap.UMAP(n_components=2,
                      random_state=42,
                      n_neighbors=30,
                      min_dist=0.3)
umap_expr = reducer.fit_transform(expr_sc)

for st in subtype_names:
    mask = subtypes == st
    axes[1].scatter(
        umap_expr[mask, 0],
        umap_expr[mask, 1],
        label=st,
        color=colors.get(st, 'gray'),
        alpha=0.7, s=20
    )
axes[1].set_title('UMAP — Gene Expression', fontsize=13)
axes[1].set_xlabel('UMAP 1')
axes[1].set_ylabel('UMAP 2')
axes[1].legend(markerscale=2, fontsize=9,
               loc='best')

# Panel 3: Expression heatmap top 50 genes
print("Generating heatmap...")
top50       = expr.var().nlargest(50).index
expr_top    = expr[top50]
sort_idx    = labels.sort_values().index
expr_ord    = expr_top.loc[sort_idx]
expr_scaled = scaler.fit_transform(expr_ord)

im = axes[2].imshow(
    expr_scaled.T,
    aspect='auto',
    cmap='RdBu_r',
    vmin=-3, vmax=3
)
axes[2].set_title('Top 50 Variable Genes\n(sorted by subtype)',
                  fontsize=13)
axes[2].set_xlabel('Patients')
axes[2].set_ylabel('Genes')
plt.colorbar(im, ax=axes[2],
             label='z-score', shrink=0.8)

plt.suptitle('TCGA-BRCA Multi-Omics EDA',
             fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig('figures/eda_overview.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/eda_overview.png")

# ── Figure 2: UMAP of methylation ───────────────────
print("Running UMAP on methylation PCs (~2 min)...")
meth_sc   = scaler.fit_transform(meth.iloc[:, :100])
umap_meth = reducer.fit_transform(meth_sc)

fig2, ax = plt.subplots(figsize=(8, 6))
for st in subtype_names:
    mask = subtypes == st
    ax.scatter(
        umap_meth[mask, 0],
        umap_meth[mask, 1],
        label=st,
        color=colors.get(st, 'gray'),
        alpha=0.7, s=20
    )
ax.set_title('UMAP — DNA Methylation (top 100 PCs)',
             fontsize=13)
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.legend(markerscale=2)
plt.tight_layout()
plt.savefig('figures/umap_methylation.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/umap_methylation.png")

# ── Figure 3: Class distribution pie ────────────────
fig3, ax3 = plt.subplots(figsize=(7, 7))
counts2    = pd.Series(subtypes).value_counts()
pie_colors = [colors.get(s, 'gray')
              for s in counts2.index]
ax3.pie(
    counts2.values,
    labels=[f'{s}\n(n={v})'
            for s, v in zip(counts2.index,
                            counts2.values)],
    colors=pie_colors,
    autopct='%1.1f%%',
    startangle=90,
    wedgeprops={'linewidth': 1,
                'edgecolor': 'white'}
)
ax3.set_title('Breast Cancer Subtype Distribution\n'
              'TCGA-BRCA PAM50', fontsize=13)
plt.tight_layout()
plt.savefig('figures/subtype_distribution.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/subtype_distribution.png")
print("\nEDA complete!")
