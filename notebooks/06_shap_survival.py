"""
notebooks/06_shap_survival.py
SHAP explainability + Kaplan-Meier survival analysis
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, sys, os
sys.path.append('.')
import shap
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from lifelines import KaplanMeierFitter

print("="*55)
print("PHASE 5: SHAP + Survival Analysis")
print("="*55)

os.makedirs('figures', exist_ok=True)

# ── 1. Load data ─────────────────────────────────────
print("\nLoading data...")
expr   = pd.read_csv('data/processed/expression_clean.csv',
                     index_col=0)
labels = pd.read_csv('data/processed/labels.csv',
                     index_col=0).squeeze()
clinical = pd.read_csv('data/raw/clinical_subtypes.csv',
                       index_col=0)

le = LabelEncoder()
y  = le.fit_transform(labels)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(expr)

X_tr, X_te, y_tr, y_te = train_test_split(
    X_scaled, y,
    test_size=0.2, random_state=42, stratify=y)

feature_names = expr.columns.tolist()
print(f"Samples: {len(labels)}")
print(f"Features: {len(feature_names)}")

# ── 2. Load the best model ───────────────────────────
print("\nLooking for saved model...")
model_files = [f for f in os.listdir('models')
               if 'expression' in f.lower()
               or 'gene' in f.lower()]
print(f"Found: {model_files}")

# Try to load whichever expression model exists
model_path = None
for candidate in [
    'models/gene_expression_xgb.pkl',
    'models/gene_expression_rf.pkl',
    'models/gene_expression_xgboost.pkl',
]:
    if os.path.exists(candidate):
        model_path = candidate
        break

if model_path is None:
    # Retrain quickly if model not found
    print("Model not found — retraining XGBoost...")
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.1,
        eval_metric='mlogloss',
        random_state=42, n_jobs=-1)
    model.fit(X_tr, y_tr)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/gene_expression_xgb.pkl')
    model_path = 'models/gene_expression_xgb.pkl'
    print("Saved: models/gene_expression_xgb.pkl")

model = joblib.load(model_path)
print(f"Loaded: {model_path}")

# ── 3. SHAP values ───────────────────────────────────
print("\nCalculating SHAP values (~2 min)...")
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(
    X_te[:150], check_additivity=False)

sv_arr = np.array(shap_values)
print(f"SHAP output shape: {sv_arr.shape}")

# ── Fix SHAP output for multi-class ─────────────────
sv_arr = np.array(shap_values)
print(f"SHAP output shape: {sv_arr.shape}")

if sv_arr.ndim == 3:
    # Shape: (n_samples, n_features, n_classes)
    # Take mean across classes
    mean_abs_shap = np.mean(np.abs(sv_arr), axis=2)
    mean_abs_shap = np.mean(mean_abs_shap, axis=0)

elif sv_arr.ndim == 2:
    # Binary classification
    mean_abs_shap = np.mean(np.abs(sv_arr), axis=0)

else:
    raise ValueError(f"Unexpected SHAP shape: {sv_arr.shape}")
# ── Select top 20 important genes ─────────────────
top20_idx   = np.argsort(mean_abs_shap)[::-1][:20]
top20_vals  = mean_abs_shap[top20_idx]
top20_names = [feature_names[i] for i in top20_idx]

print("\nTop 10 genes driving subtype prediction:")
for i, (name, val) in enumerate(zip(top20_names[:10], top20_vals[:10])):
    print(f"  {i+1:2d}. {name:<25} SHAP={val:.4f}")
# ── 4. Plot SHAP bar chart ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

colors_bar = plt.cm.RdPu(
    np.linspace(0.4, 0.9, 20))[::-1]
axes[0].barh(
    range(20), top20_vals[::-1],
    color=colors_bar)
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(
    top20_names[::-1], fontsize=9)
axes[0].set_xlabel('Mean |SHAP value|')
axes[0].set_title(
    'Top 20 Genes Driving Subtype Prediction\n'
    '(Gene Expression — XGBoost)',
    fontsize=12)
axes[0].grid(True, alpha=0.3, axis='x')

# ── 5. Kaplan-Meier survival curves ─────────────────
print("\nGenerating Kaplan-Meier survival curves...")

# Prepare clinical data
clinical = clinical.rename(columns={
    'paper_BRCA_Subtype_PAM50': 'subtype'})

# Get survival time
def get_survival(row):
    if row.get('vital_status') == 'Dead':
        return row.get('days_to_death', np.nan)
    return row.get('days_to_last_follow_up', np.nan)

clinical['survival_days'] = clinical.apply(
    get_survival, axis=1)
clinical['event'] = (
    clinical['vital_status'] == 'Dead').astype(int)
clinical['survival_years'] = (
    clinical['survival_days'] / 365.25)

clinical_clean = clinical.dropna(
    subset=['survival_days', 'subtype',
            'survival_years'])
clinical_clean = clinical_clean[
    clinical_clean['survival_days'] > 0]
clinical_clean = clinical_clean[
    clinical_clean['subtype'] != 'Normal']

print(f"Clinical samples for survival: {len(clinical_clean)}")
print(f"Subtypes: {clinical_clean['subtype'].value_counts().to_dict()}")

subtype_colors = {
    'LumA':  '#378ADD',
    'LumB':  '#1D9E75',
    'Her2':  '#D85A30',
    'Basal': '#E24B4A'
}

kmf = KaplanMeierFitter()
ax  = axes[1]

for subtype, color in subtype_colors.items():
    mask = clinical_clean['subtype'] == subtype
    n    = mask.sum()
    if n < 10:
        print(f"Skipping {subtype} — only {n} samples")
        continue
    kmf.fit(
        clinical_clean.loc[mask, 'survival_years'],
        clinical_clean.loc[mask, 'event'],
        label=f"{subtype} (n={n})"
    )
    kmf.plot_survival_function(
        ax=ax, color=color, linewidth=2)

ax.set_xlabel('Time (years)', fontsize=11)
ax.set_ylabel('Survival probability', fontsize=11)
ax.set_title(
    'Kaplan-Meier Survival by PAM50 Subtype\n'
    'TCGA-BRCA Clinical Validation',
    fontsize=12)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9, loc='lower left')

plt.suptitle(
    'SHAP Explainability + Clinical Survival Validation',
    fontsize=14)
plt.tight_layout()
plt.savefig('figures/shap_survival.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/shap_survival.png")

# ── 6. Save top genes to CSV ─────────────────────────
top_genes_df = pd.DataFrame({
    'gene': top20_names,
    'mean_shap': top20_vals
})
top_genes_df.to_csv('logs/top_genes_shap.csv',
                    index=False)
print("Saved: logs/top_genes_shap.csv")

print("\n" + "="*55)
print("PHASE 5 COMPLETE")
print("="*55)
print("Figures saved:")
print("  figures/shap_survival.png")
print("Top genes saved:")
print("  logs/top_genes_shap.csv")
