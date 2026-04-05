"""
notebooks/04_single_omics.py
Single-omics cancer subtype classification
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import (StandardScaler,
    LabelEncoder)
from sklearn.metrics import (accuracy_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay)
import xgboost as xgb
import shap

print("="*55)
print("PHASE 3: Single-omics classification")
print("="*55)

os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)
os.makedirs('figures', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ── Load data ────────────────────────────────────────
expr  = pd.read_csv('data/processed/expression_clean.csv',
                    index_col=0)
meth  = pd.read_csv('data/processed/methylation_pca.csv',
                    index_col=0)
cnv   = pd.read_csv('data/processed/cnv_clean.csv',
                    index_col=0)
labels = pd.read_csv('data/processed/labels.csv',
                     index_col=0).squeeze()

le = LabelEncoder()
y = le.fit_transform(labels)
print(f"Classes: {list(le.classes_)}")
print(f"Samples: {len(y)}")

omics = {
    'Gene Expression': expr,
    'DNA Methylation': meth,
    'CNV':            cnv
}

results = []
skf = StratifiedKFold(n_splits=5,
                      shuffle=True, random_state=42)

for name, X in omics.items():
    print(f"\n{'─'*45}")
    print(f"Training on: {name} ({X.shape[1]} features)")
    print(f"{'─'*45}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Random Forest ───────────────────────────────
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=None,
        class_weight='balanced',
        random_state=42, n_jobs=-1)
    rf_cv = cross_val_score(
        rf, X_scaled, y, cv=skf,
        scoring='accuracy', n_jobs=-1)
    print(f"RF  CV Accuracy: "
          f"{rf_cv.mean():.4f} ± {rf_cv.std():.4f}")

    # ── XGBoost ─────────────────────────────────────
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=5,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42, n_jobs=-1)
    xgb_cv = cross_val_score(
        xgb_clf, X_scaled, y, cv=skf,
        scoring='accuracy', n_jobs=-1)
    print(f"XGB CV Accuracy: "
          f"{xgb_cv.mean():.4f} ± {xgb_cv.std():.4f}")

    # ── Train best model on full data ───────────────
    best_model = (rf if rf_cv.mean() > xgb_cv.mean()
                  else xgb_clf)
    best_name  = ('RF' if rf_cv.mean() > xgb_cv.mean()
                  else 'XGB')
    best_acc   = max(rf_cv.mean(), xgb_cv.mean())

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_scaled, y, test_size=0.2,
        random_state=42, stratify=y)
    best_model.fit(X_tr, y_tr)
    y_pred = best_model.predict(X_te)
    test_acc = accuracy_score(y_te, y_pred)

    print(f"Test Accuracy ({best_name}): {test_acc:.4f}")
    print(classification_report(
        y_te, y_pred,
        target_names=le.classes_))

    # Save model
    safe_name = name.replace(' ', '_').lower()
    joblib.dump(best_model,
        f'models/{safe_name}_{best_name.lower()}.pkl')

    # Confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(
        cm, display_labels=le.classes_)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f'Confusion Matrix — {name}')
    plt.tight_layout()
    plt.savefig(
        f'figures/cm_{safe_name}.png', dpi=150)
    plt.close()

    results.append({
        'omics': name,
        'best_model': best_name,
        'cv_accuracy': round(best_acc, 4),
        'test_accuracy': round(test_acc, 4)
    })

# ── Summary ──────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv('logs/single_omics_results.csv',
                  index=False)
print("\n" + "="*55)
print("SINGLE-OMICS RESULTS SUMMARY")
print("="*55)
print(results_df.to_string(index=False))

# ── Bar chart of accuracies ──────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = range(len(results_df))
bars = ax.bar(x, results_df['cv_accuracy'],
    color=['#378ADD','#1D9E75','#D85A30'], width=0.5)
ax.set_xticks(x)
ax.set_xticklabels(results_df['omics'])
ax.set_ylim(0.5, 1.0)
ax.set_ylabel('5-fold CV Accuracy')
ax.set_title('Single-Omics Model Comparison')
ax.grid(True, alpha=0.3, axis='y')
for bar, acc in zip(bars, results_df['cv_accuracy']):
    ax.text(bar.get_x()+bar.get_width()/2,
        bar.get_height()+0.005,
        f'{acc:.3f}', ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('figures/single_omics_comparison.png',
            dpi=150)
plt.close()
print("Saved: figures/single_omics_comparison.png")
print("\nSingle-omics training complete!")
