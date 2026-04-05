"""
notebooks/05_multi_omics_fusion.py
Multi-omics fusion for cancer subtype classification
Early fusion and late fusion strategies
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib, os, sys
sys.path.append('.')
from sklearn.model_selection import (StratifiedKFold,
    cross_val_score, train_test_split)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay)
from sklearn.pipeline import Pipeline
import xgboost as xgb

print("="*55)
print("PHASE 4: Multi-omics fusion")
print("="*55)

# ── Load data ────────────────────────────────────────
expr   = pd.read_csv('data/processed/expression_clean.csv',
                     index_col=0)
meth   = pd.read_csv('data/processed/methylation_pca.csv',
                     index_col=0)
cnv    = pd.read_csv('data/processed/cnv_clean.csv',
                     index_col=0)
labels = pd.read_csv('data/processed/labels.csv',
                     index_col=0).squeeze()

le = LabelEncoder()
y  = le.fit_transform(labels)

scaler = StandardScaler()
expr_s = scaler.fit_transform(expr)
meth_s = scaler.fit_transform(meth)
cnv_s  = scaler.fit_transform(cnv)

skf = StratifiedKFold(n_splits=5,
                      shuffle=True, random_state=42)
results = []

# ── Strategy 1: Early fusion ─────────────────────────
print("\n" + "─"*45)
print("Strategy 1: Early Fusion")
print("Concatenate all 3 omics → train one model")
print("─"*45)

X_early = np.hstack([expr_s, meth_s, cnv_s])
print(f"Combined feature matrix: {X_early.shape}")

xgb_early = xgb.XGBClassifier(
    n_estimators=300, max_depth=6,
     learning_rate=0.05, subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric='mlogloss',
    random_state=42, n_jobs=-1)

early_cv = cross_val_score(
    xgb_early, X_early, y,
    cv=skf, scoring='accuracy', n_jobs=-1)
print(f"Early Fusion CV Accuracy: "
      f"{early_cv.mean():.4f} ± {early_cv.std():.4f}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_early, y, test_size=0.2,
    random_state=42, stratify=y)
xgb_early.fit(X_tr, y_tr)
early_test_acc = accuracy_score(y_te,
                                xgb_early.predict(X_te))
print(f"Early Fusion Test Accuracy: {early_test_acc:.4f}")
joblib.dump(xgb_early, 'models/early_fusion_xgb.pkl')

results.append({
    'strategy': 'Early Fusion',
    'cv_accuracy': round(early_cv.mean(), 4),
    'test_accuracy': round(early_test_acc, 4)
})

# ── Strategy 2: Late fusion ──────────────────────────
print("\n" + "─"*45)
print("Strategy 2: Late Fusion")
print("Train 3 models → average probabilities")
print("─"*45)

X_tr_e, X_te_e, y_tr, y_te = train_test_split(
    expr_s, y, test_size=0.2,
    random_state=42, stratify=y)
X_tr_m = meth_s[
    [i for i in range(len(y)) if i not in
     list(range(int(len(y)*0.8), len(y)))]
]
X_te_m = meth_s[
    list(range(int(len(y)*0.8), len(y)))
]
X_tr_c = cnv_s[
    [i for i in range(len(y)) if i not in
     list(range(int(len(y)*0.8), len(y)))]
]
X_te_c = cnv_s[
    list(range(int(len(y)*0.8), len(y)))
]

# Use proper index-based splitting
idx = np.arange(len(y))
np.random.seed(42)
from sklearn.model_selection import train_test_split
tr_idx, te_idx = train_test_split(
    idx, test_size=0.2, random_state=42, stratify=y)

y_tr_late = y[tr_idx]
y_te_late = y[te_idx]

m1 = xgb.XGBClassifier(n_estimators=200,
    use_label_encoder=False,
    eval_metric='mlogloss', random_state=42)
m2 = xgb.XGBClassifier(n_estimators=200,
    use_label_encoder=False,
    eval_metric='mlogloss', random_state=42)
m3 = xgb.XGBClassifier(n_estimators=200,
    use_label_encoder=False,
    eval_metric='mlogloss', random_state=42)

m1.fit(expr_s[tr_idx], y_tr_late)
m2.fit(meth_s[tr_idx], y_tr_late)
m3.fit(cnv_s[tr_idx],  y_tr_late)

prob1 = m1.predict_proba(expr_s[te_idx])
prob2 = m2.predict_proba(meth_s[te_idx])
prob3 = m3.predict_proba(cnv_s[te_idx])

avg_probs = (prob1 + prob2 + prob3) / 3
late_preds = np.argmax(avg_probs, axis=1)
late_acc = accuracy_score(y_te_late, late_preds)
print(f"Late Fusion Test Accuracy: {late_acc:.4f}")

results.append({
    'strategy': 'Late Fusion',
    'cv_accuracy': late_acc,
    'test_accuracy': late_acc
})

# ── Load single-omics results for comparison ─────────
single = pd.read_csv('logs/single_omics_results.csv')
for _, row in single.iterrows():
    results.append({
        'strategy': row['omics'],
        'cv_accuracy': row['cv_accuracy'],
        'test_accuracy': row['test_accuracy']
    })

results_df = pd.DataFrame(results)
results_df.to_csv('logs/all_results.csv', index=False)

# ── Final comparison plot ────────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
colors_bar = ['#1D9E75','#0F6E56',
              '#378ADD','#7F77DD','#D85A30']
bars = ax.bar(results_df['strategy'],
    results_df['test_accuracy'],
    color=colors_bar[:len(results_df)], width=0.5)
ax.set_ylim(0.5, 1.05)
ax.set_ylabel('Test Accuracy')
ax.set_title('Multi-Omics vs Single-Omics: '
             'Cancer Subtype Classification')
ax.grid(True, alpha=0.3, axis='y')
plt.xticks(rotation=20, ha='right')
for bar, acc in zip(bars, results_df['test_accuracy']):
    ax.text(bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.005,
        f'{acc:.3f}', ha='center', fontsize=11)
plt.tight_layout()
plt.savefig('figures/all_models_comparison.png',
            dpi=150, bbox_inches='tight')
plt.close()
print("Saved: figures/all_models_comparison.png")

print("\n" + "="*55)
print("ALL RESULTS SUMMARY")
print("="*55)
print(results_df.to_string(index=False))
print("\nMulti-omics fusion complete!")
