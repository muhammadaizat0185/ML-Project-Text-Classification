import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, classification_report

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    Computes standard evaluation metrics alongside critical metrics for academic integrity
    (False Positive Rate and Specificity) to minimize false accusations.
    """
    # Confusion Matrix: TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Core calculations
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    # Specificity = TN / (TN + FP) -> proportion of true human essays correctly classified
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    # False Positive Rate = FP / (FP + TN) -> proportion of human essays falsely accused
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    
    # Precision = TP / (TP + FP) -> proportion of flagged essays that are actually AI
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    
    # Recall (Sensitivity) = TP / (TP + FN) -> proportion of AI essays detected
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    auc = roc_auc_score(y_true, y_prob) if y_prob is not None else None
    
    metrics = {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'false_positive_rate': fpr,
        'auc_roc': auc,
        'confusion_matrix': {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
    }
    return metrics

def print_evaluation_report(model_name, metrics):
    """
    Prints a detailed, highly readable terminal report of a model's performance.
    """
    print("=" * 60)
    print(f" EVALUATION REPORT FOR: {model_name.upper()} ")
    print("=" * 60)
    print(f"Accuracy:                 {metrics['accuracy']*100:.3f}%")
    print(f"F1-Score:                 {metrics['f1_score']:.4f}")
    if metrics['auc_roc'] is not None:
        print(f"ROC-AUC:                  {metrics['auc_roc']:.4f}")
    
    print("\n--- Stylistic & Academic Integrity Safety Metrics ---")
    print(f"Precision (Positive Predictive Value):  {metrics['precision']*100:.3f}%")
    print(f"Recall / Sensitivity (Detection Rate):  {metrics['recall']*100:.3f}%")
    print(f"Specificity (True Negative Rate):       {metrics['specificity']*100:.3f}%")
    
    # Highlight False Positive Rate (False Accusations)
    fpr_pct = metrics['false_positive_rate'] * 100
    print("-" * 60)
    if fpr_pct > 1.0:
        print(f"[!] False Positive Rate (Falsely Accused): {fpr_pct:.3f}%  (HIGH RISK!)")
    elif fpr_pct > 0.1:
        print(f"[!] False Positive Rate (Falsely Accused): {fpr_pct:.3f}%  (MODERATE RISK)")
    else:
        print(f"[OK] False Positive Rate (Falsely Accused): {fpr_pct:.3f}%  (SAFE / LOW RISK)")
    print("-" * 60)
    
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"  Predicted Human  |  TN: {cm['tn']}  |  FN: {cm['fn']}")
    print(f"  Predicted AI     |  FP: {cm['fp']}  |  TP: {cm['tp']}")
    print("=" * 60)

def tune_decision_threshold(y_true, y_prob, target_fpr=0.001):
    """
    Scans different probability thresholds to find the threshold that guarantees 
    the False Positive Rate (FPR) is at or below the 'target_fpr'.
    Locks down the threshold to protect innocent students.
    """
    print(f"\nScanning thresholds to enforce False Positive Rate <= {target_fpr*100:.3f}%...")
    
    # Sort probabilities
    thresholds = np.linspace(0.0, 1.0, 1001)
    best_threshold = 0.5
    best_metrics = None
    found = False
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        # We want the threshold that yields FPR <= target_fpr while maximizing recall
        if metrics['false_positive_rate'] <= target_fpr:
            if not found or metrics['recall'] > best_metrics['recall']:
                best_threshold = t
                best_metrics = metrics
                found = True
                
    if found:
        print(f"[OK] Found Optimal Academic Integrity Threshold: {best_threshold:.3f}")
        print(f"   Recall (Detection Rate) at this threshold: {best_metrics['recall']*100:.2f}%")
        print(f"   Actual False Positive Rate achieved:       {best_metrics['false_positive_rate']*100:.3f}%")
        return best_threshold, best_metrics
    else:
        print("[!] Could not satisfy the exact FPR target with current model predictions. Using fallback threshold of 0.99.")
        fallback_pred = (y_prob >= 0.99).astype(int)
        return 0.99, calculate_metrics(y_true, fallback_pred, y_prob)


def plot_global_evaluation(y_true, y_prob, threshold, model_name, filename_prefix):
    """
    Plots the global Out-of-Fold ROC Curve (with AUC) and Confusion Matrix (tuned threshold)
    and saves them to the results/ folder.
    """
    import os
    from sklearn.metrics import roc_curve, confusion_matrix, auc
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Create results dir if not exists
    os.makedirs("results", exist_ok=True)
    
    # 1. Confusion Matrix
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human', 'AI'], yticklabels=['Human', 'AI'])
    plt.title(f"{model_name} Confusion Matrix\n(Tuned Threshold = {threshold:.3f})")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    cm_path = f"results/{filename_prefix}_confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve (Global OOF)')
    plt.legend(loc="lower right")
    plt.tight_layout()
    roc_path = f"results/{filename_prefix}_roc_curve.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()
    
    print(f"Saved evaluation plots for {model_name} to results/")


if __name__ == "__main__":
    # Quick sanity test with dummy data
    y_test = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95])
    y_pred = (y_prob >= 0.5).astype(int)
    
    dummy_metrics = calculate_metrics(y_test, y_pred, y_prob)
    print_evaluation_report("Sanity Check Model", dummy_metrics)
    
    best_t, best_m = tune_decision_threshold(y_test, y_prob, target_fpr=0.25)
    print_evaluation_report("Tuned Model (FPR <= 25%)", best_m)
