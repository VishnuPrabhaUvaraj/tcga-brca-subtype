"""
notebooks/02_preprocessing.py
Load, clean and align all 3 omics datasets
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os, warnings
warnings.filterwarnings('ignore')

print("="*55)
print("PREPROCESSING: Aligning 3 omics datasets")
print("="*55)

os.makedirs('data/processed', exist_ok=True)

# ── 1. Load clinical + subtype labels ───────────────
print("\nLoading clinical data...")
clinical = pd.read_csv('data/raw/clinical_subtypes.csv',
                       index_col=0)
clinical.columns = clinical.columns.str.strip()
clinical = clinical.rename(columns={
    'paper_BRCA_Subtype_PAM50': 'subtype'})
clinical = clinical.dropna(subset=['subtype'])
clinical = clinical[clinical['subtype'] != 'Normal']
# Shorten barcode to 12 chars (patient ID level)
clinical.index = clinical['barcode'].str[:12]
print(f"Clinical samples: {len(clinical)}")
print(clinical['subtype'].value_counts())

# ── 2. Load gene expression ──────────────────────────
print("\nLoading gene expression...")
expr = pd.read_csv('data/raw/expression_raw.csv',
                   index_col=0)
expr = expr.T
# Shorten to 12 chars
expr.index = expr.index.str[:12]
# Remove duplicate patient IDs
expr = expr[~expr.index.duplicated(keep='first')]
print(f"Expression shape: {expr.shape}")

expr = np.log2(expr + 1)
gene_var  = expr.var()
top_genes = gene_var.nlargest(5000).index
expr      = expr[top_genes]
print(f"After variance filter: {expr.shape}")

# ── 3. Load methylation ──────────────────────────────
print("\nLoading methylation (5-10 min)...")
meth = pd.read_csv('data/raw/methylation_raw.csv',
                   index_col=0)
meth = meth.T
meth.index = meth.index.str[:12]
meth = meth[~meth.index.duplicated(keep='first')]
print(f"Methylation shape: {meth.shape}")

meth = meth.fillna(meth.mean())

print("Running PCA (→ 500 PCs)...")
scaler_meth = StandardScaler()
meth_scaled = scaler_meth.fit_transform(meth)
n_pcs       = min(500, meth.shape[0]-1, meth.shape[1])
pca_meth    = PCA(n_components=n_pcs, random_state=42)
meth_pca    = pca_meth.fit_transform(meth_scaled)
meth_df     = pd.DataFrame(
    meth_pca,
    index=meth.index,
    columns=[f'meth_PC{i+1}' for i in range(n_pcs)]
)
var_exp = pca_meth.explained_variance_ratio_.sum()
print(f"PCA: {n_pcs} components, {var_exp:.2%} variance")

# ── 4. Load CNV ──────────────────────────────────────
print("\nLoading CNV data...")
cnv = pd.read_csv('data/raw/cnv_raw.csv', index_col=0)
cnv = cnv.T
cnv.index = cnv.index.str[:12]
cnv = cnv[~cnv.index.duplicated(keep='first')]
print(f"CNV shape: {cnv.shape}")
cnv = cnv.fillna(0)

# ── 5. Find common samples ───────────────────────────
print("\nAligning samples...")
print(f"Clinical: {len(clinical)}")
print(f"Expression: {len(expr)}")
print(f"Methylation: {len(meth_df)}")
print(f"CNV: {len(cnv)}")

common = list(
    set(clinical.index) &
    set(expr.index)     &
    set(meth_df.index)  &
    set(cnv.index)
)
print(f"Common samples: {len(common)}")

expr_aligned = expr.loc[common]
meth_aligned = meth_df.loc[common]
cnv_aligned  = cnv.loc[common]
labels       = clinical.loc[common, 'subtype']

# ── 6. Encode labels ─────────────────────────────────
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y  = le.fit_transform(labels)
print(f"\nFinal classes: {list(le.classes_)}")
for cls, cnt in zip(le.classes_, np.bincount(y)):
    print(f"  {cls}: {cnt}")

# ── 7. Save ──────────────────────────────────────────
expr_aligned.to_csv('data/processed/expression_clean.csv')
meth_aligned.to_csv('data/processed/methylation_pca.csv')
cnv_aligned.to_csv('data/processed/cnv_clean.csv')
labels.to_csv('data/processed/labels.csv')
pd.DataFrame({'subtype': le.classes_}).to_csv(
    'data/processed/label_encoder.csv', index=False)

print("\n" + "="*55)
print("PREPROCESSING COMPLETE")
print("="*55)
print(f"Expression:  {expr_aligned.shape}")
print(f"Methylation: {meth_aligned.shape}")
print(f"CNV:         {cnv_aligned.shape}")
print(f"Labels:      {labels.value_counts().to_dict()}")
