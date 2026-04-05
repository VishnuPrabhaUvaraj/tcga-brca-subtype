"""
notebooks/02_preprocessing_v2.py
Fixed alignment — ensures all datasets have identical samples
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import os, warnings
warnings.filterwarnings('ignore')

print("="*55)
print("PREPROCESSING v2: Fixed alignment")
print("="*55)

os.makedirs('data/processed', exist_ok=True)

# ── 1. Load clinical ─────────────────────────────────
print("\nStep 1: Loading clinical data...")
clinical = pd.read_csv('data/raw/clinical_subtypes.csv',
                       index_col=0)
clinical = clinical.rename(columns={
    'paper_BRCA_Subtype_PAM50': 'subtype'})
clinical = clinical.dropna(subset=['subtype'])
clinical = clinical[clinical['subtype'] != 'Normal']
clinical.index = clinical['barcode'].str[:12]
clinical = clinical[~clinical.index.duplicated(keep='first')]
print(f"Clinical: {len(clinical)} patients")

# ── 2. Load expression ───────────────────────────────
print("\nStep 2: Loading gene expression...")
expr = pd.read_csv('data/raw/expression_raw.csv',
                   index_col=0)
expr = expr.T
expr.index = expr.index.str[:12]
expr = expr[~expr.index.duplicated(keep='first')]
expr = np.log2(expr + 1)
top_genes = expr.var().nlargest(5000).index
expr = expr[top_genes]
print(f"Expression: {expr.shape}")

# ── 3. Load methylation ──────────────────────────────
print("\nStep 3: Loading methylation...")
meth = pd.read_csv('data/raw/methylation_raw.csv',
                   index_col=0)
meth = meth.T
meth.index = meth.index.str[:12]
meth = meth[~meth.index.duplicated(keep='first')]
meth = meth.fillna(meth.mean())
print(f"Methylation before PCA: {meth.shape}")

print("Running PCA (500 components)...")
sc = StandardScaler()
meth_sc = sc.fit_transform(meth)
n_pcs = min(500, meth.shape[0]-1, meth.shape[1])
pca = PCA(n_components=n_pcs, random_state=42)
meth_pca = pca.fit_transform(meth_sc)
meth_df = pd.DataFrame(
    meth_pca, index=meth.index,
    columns=[f'meth_PC{i+1}' for i in range(n_pcs)])
print(f"Methylation PCA: {meth_df.shape}")
print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

# ── 4. Load CNV ──────────────────────────────────────
print("\nStep 4: Loading CNV...")
cnv = pd.read_csv('data/raw/cnv_raw.csv', index_col=0)
cnv = cnv.T
cnv.index = cnv.index.str[:12]
cnv = cnv[~cnv.index.duplicated(keep='first')]
cnv = cnv.fillna(0)
print(f"CNV: {cnv.shape}")

# ── 5. Find common samples ───────────────────────────
print("\nStep 5: Finding common samples...")
print(f"  Clinical index size:    {len(clinical)}")
print(f"  Expression index size:  {len(expr)}")
print(f"  Methylation index size: {len(meth_df)}")
print(f"  CNV index size:         {len(cnv)}")

common = sorted(
    set(clinical.index) &
    set(expr.index)     &
    set(meth_df.index)  &
    set(cnv.index)
)
print(f"  Common samples:         {len(common)}")

# ── 6. Align ALL datasets to exact same samples ──────
print("\nStep 6: Aligning all datasets...")
expr_al  = expr.loc[common]
meth_al  = meth_df.loc[common]
cnv_al   = cnv.loc[common]
# CRITICAL: get labels ONLY for common samples
labels   = clinical.loc[common, 'subtype']

# Final check
assert len(expr_al) == len(meth_al) == \
       len(cnv_al)  == len(labels), \
    "Sample mismatch after alignment!"
print(f"All datasets aligned to {len(common)} samples")

# ── 7. Encode labels ─────────────────────────────────
le = LabelEncoder()
y  = le.fit_transform(labels)
print(f"\nClasses: {list(le.classes_)}")
for cls, cnt in zip(le.classes_, np.bincount(y)):
    print(f"  {cls}: {cnt}")

# ── 8. Save ──────────────────────────────────────────
print("\nStep 8: Saving processed files...")
expr_al.to_csv('data/processed/expression_clean.csv')
meth_al.to_csv('data/processed/methylation_pca.csv')
cnv_al.to_csv('data/processed/cnv_clean.csv')
labels.to_csv('data/processed/labels.csv')
pd.DataFrame({'subtype': le.classes_}).to_csv(
    'data/processed/label_encoder.csv', index=False)

print("\n" + "="*55)
print("PREPROCESSING COMPLETE")
print("="*55)
print(f"Expression:  {expr_al.shape}")
print(f"Methylation: {meth_al.shape}")
print(f"CNV:         {cnv_al.shape}")
print(f"Labels:      {len(labels)}")
print(f"Subtype counts: {labels.value_counts().to_dict()}")
