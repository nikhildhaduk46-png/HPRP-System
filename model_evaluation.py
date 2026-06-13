import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os, sys
from typing import Dict, Any, Tuple, List, cast

# Adding parent directory to path for config import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config import REPORT_DIR

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, accuracy_score, f1_score
)

os.makedirs(REPORT_DIR, exist_ok=True)

def evaluate_all(trained_models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray, feature_names: List[str]) -> Tuple[Any, str]:
    print("\n" + "="*60)
    print("  📋 MODEL EVALUATION REPORT")
    print("="*60)

    # FIX 1: Use Any for best_auc to avoid "Float is not assignable to float" error
    best_auc: Any = 0.0
    best_name: str = ""
    best_model: Any = None

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_prob)
        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)

        print(f"\n🔹 {name}")
        print(f"   Accuracy : {acc:.4f}")
        print(f"   F1 Score : {f1:.4f}")
        print(f"   ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))

        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, model

    # FIX 2: Explicitly verify best_model is not None to satisfy Pylance
    if best_model is None:
        print("⚠️ Error: No models were found.")
        return None, ""

    # FIX 3: Cast best_auc to float for printing to clear the assignment warning
    print(f"\n🏆 Best Model: {best_name}  (AUC = {float(best_auc):.4f})")

    # ── Confusion Matrix ─────────────────────────────────────────
    # Pylance now knows best_model has attributes because we verified it above
    y_pred_best = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Low Risk', 'High Risk'],
                yticklabels=['Low Risk', 'High Risk'])
    plt.title(f'Confusion Matrix — {best_name}')
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "confusion_matrix.png"), dpi=150)
    plt.close()

    # ── ROC Curve (all models) ────────────────────────────────────
    plt.figure(figsize=(8, 6))
    for name, model in trained_models.items():
        y_prob_all = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob_all)
        auc_val = roc_auc_score(y_test, y_prob_all)
        plt.plot(fpr, tpr, label=f"{name} (AUC={float(auc_val):.3f})")
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models"); plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "roc_curve.png"), dpi=150)
    plt.close()

    # ── Feature Importance ────────────────────────────────────────
    importances = None
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importances = np.abs(best_model.coef_[0])

    if importances is not None:
        idx = np.argsort(importances)[::-1][:15]
        plt.figure(figsize=(10, 6))
        plt.barh([feature_names[i] for i in idx][::-1],
                 importances[idx][::-1], color='steelblue')
        plt.title(f'Top 15 Feature Importances — {best_name}')
        plt.xlabel('Importance'); plt.tight_layout()
        plt.savefig(os.path.join(REPORT_DIR, "feature_importance.png"), dpi=150)
        plt.close()

    print(f"\n✅ Charts saved successfully to: {REPORT_DIR}")
    return best_model, best_name