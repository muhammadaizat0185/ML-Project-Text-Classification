# -*- coding: utf-8 -*-
"""
Member 4 - Random Forest Model
Task: AI-Generated vs Human-Written Text Classification
Pipeline: Load preprocessed K-Fold data → Train → Evaluate & Tune Threshold
"""

import os
import sys
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.sparse import load_npz
from sklearn.ensemble import RandomForestClassifier

# Import the shared evaluation utilities
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from evaluate import calculate_metrics, print_evaluation_report, tune_decision_threshold


# =============================================================================
# CONFIGURATION

# =============================================================================

PROCESSED_DATA_DIR = "processed_data"   # Output from preprocess.py
MODELS_DIR         = "models"           # Where vectorizers were saved
OUTPUT_DIR         = "results"          # Where this script saves outputs
N_FOLDS            = 5
MODEL_NAME         = "Random Forest"

# Target False Positive Rate ceiling (academic integrity safety threshold)
TARGET_FPR = 0.001   # 0.1% — minimise false accusations of students


# =============================================================================
# HELPERS

# =============================================================================

def load_fold_data(fold: int):
    """Load preprocessed sparse matrices and labels for a given fold."""
    fold_dir = os.path.join(PROCESSED_DATA_DIR, f"fold_{fold}")
    
    required = ["X_train.npz", "X_test.npz", "y_train.npy", "y_test.npy"]
    for f in required:
        path = os.path.join(fold_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing file: {path}\n"
                "Please run preprocess.py first to generate the fold data."
            )
    
    X_train = load_npz(os.path.join(fold_dir, "X_train.npz"))
    X_test  = load_npz(os.path.join(fold_dir, "X_test.npz"))
    y_train = np.load(os.path.join(fold_dir, "y_train.npy"))
    y_test  = np.load(os.path.join(fold_dir, "y_test.npy"))
    
    return X_train, X_test, y_train, y_test


def save_fold_plots(fold: int, y_true, y_pred, feature_importances=None, feature_names=None):
    """Generate and save evaluation plots for the fold instead of blocking GUI."""
    fold_results_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    os.makedirs(fold_results_dir, exist_ok=True)
    
    # 1. Confusion Matrix Plot
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Human', 'AI'],
        yticklabels=['Human', 'AI']
    )
    plt.title(f"Random Forest Confusion Matrix - Fold {fold}")
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.tight_layout()
    cm_plot_path = os.path.join(fold_results_dir, "rf_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=150)
    plt.close()
    
    # 2. Feature Importance Plot (if available)
    if feature_importances is not None and feature_names is not None:
        top_n = 20
        indices = np.argsort(feature_importances)[-top_n:]
        plt.figure(figsize=(10, 6))
        plt.barh(
            range(top_n),
            feature_importances[indices]
        )
        plt.yticks(
            range(top_n),
            [feature_names[i] for i in indices]
        )
        plt.xlabel("Importance Score")
        plt.ylabel("Features")
        plt.title(f"Top 20 Important Features - Random Forest - Fold {fold}")
        plt.tight_layout()
        fi_plot_path = os.path.join(fold_results_dir, "rf_feature_importance.png")
        plt.savefig(fi_plot_path, dpi=150)
        plt.close()


def save_fold_results(fold: int, model, metrics_default: dict,
                      metrics_tuned: dict, best_threshold: float):
    """Persist the trained model and metrics for this fold."""
    fold_model_dir = os.path.join(MODELS_DIR, f"fold_{fold}")
    os.makedirs(fold_model_dir, exist_ok=True)
    
    # Save RF model
    model_path = os.path.join(fold_model_dir, "random_forest_model.joblib")
    joblib.dump(model, model_path)
    
    # Save metrics as JSON
    fold_results_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    os.makedirs(fold_results_dir, exist_ok=True)
    
    result_record = {
        "fold"            : fold,
        "best_threshold"  : best_threshold,
        "metrics_default" : {k: v for k, v in metrics_default.items() if k != "confusion_matrix"},
        "metrics_tuned"   : {k: v for k, v in metrics_tuned.items()   if k != "confusion_matrix"},
        "confusion_matrix_default": metrics_default["confusion_matrix"],
        "confusion_matrix_tuned"  : metrics_tuned["confusion_matrix"],
    }
    
    json_path = os.path.join(fold_results_dir, "rf_results.json")
    with open(json_path, "w") as f:
        json.dump(result_record, f, indent=4)
    
    print(f"  Model saved   -> {model_path}")
    print(f"  Metrics saved -> {json_path}")



# =============================================================================
# MAIN TRAINING LOOP

# =============================================================================

def run_training():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 65)
    print(f"  MODEL: {MODEL_NAME}")
    print(f"  Folds : {N_FOLDS}  |  Target FPR ceiling: {TARGET_FPR*100:.1f}%")
    print("=" * 65)
    
    # Accumulators for cross-fold summary
    fold_summaries = []
    
    # Global OOF prediction accumulators
    oof_probs = []
    oof_labels = []
    
    for fold in range(N_FOLDS):
        print(f"\n{'='*65}")
        print(f"  FOLD {fold} / {N_FOLDS - 1}")
        print(f"{'='*65}")
        
        # 1. Load preprocessed data
        print(f"\n[1/5] Loading fold {fold} data...")
        X_train, X_test, y_train, y_test = load_fold_data(fold)
        print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
        print(f"  y_train - AI: {y_train.sum()}  Human: {(y_train==0).sum()}")
        
        # 2. Train Random Forest model
        # Using optimized parameters for fast execution and memory safety on sparse data
        print(f"\n[2/5] Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=150,
            max_depth=20,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        print("  Training Complete!")
        
        # 3. Predict probabilities
        print(f"\n[3/5] Evaluating fold predictions...")
        y_prob_test = model.predict_proba(X_test)[:, 1]
        y_pred_default = (y_prob_test >= 0.5).astype(int)
        
        oof_probs.append(y_prob_test)
        oof_labels.append(y_test)
        
        # Calculate fold metrics (at default threshold 0.5)
        metrics_default = calculate_metrics(y_test, y_pred_default, y_prob_test)
        print_evaluation_report(f"{MODEL_NAME} - Fold {fold} (threshold=0.50)", metrics_default)
        
        # Tune decision threshold for individual fold (optional / comparison)
        best_threshold, metrics_tuned = tune_decision_threshold(y_test, y_prob_test, target_fpr=TARGET_FPR)
        print_evaluation_report(
            f"{MODEL_NAME} - Fold {fold} (threshold={best_threshold:.3f}, FPR-tuned)",
            metrics_tuned
        )
        
        # 4. Generate and save plots
        print(f"\n[4/5] Generating evaluation plots...")
        # Get feature names from saved vectorizers for importance plot
        try:
            fold_models_dir = os.path.join(MODELS_DIR, f"fold_{fold}")
            word_vect = joblib.load(os.path.join(fold_models_dir, "word_vectorizer.joblib"))
            char_vect = joblib.load(os.path.join(fold_models_dir, "char_vectorizer.joblib"))
            feature_names = list(word_vect.get_feature_names_out()) + list(char_vect.get_feature_names_out())
        except Exception as e:
            print(f"  Warning: Could not load vectorizers for feature names: {e}")
            feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
            
        save_fold_plots(fold, y_test, y_pred_default, model.feature_importances_, feature_names)
        
        # 5. Persist model + metrics
        print(f"\n[5/5] Saving model and results...")
        save_fold_results(fold, model, metrics_default, metrics_tuned, best_threshold)
        
        fold_summaries.append({
            "fold"           : fold,
            "best_threshold" : best_threshold,
            "default"        : metrics_default,
            "tuned"          : metrics_tuned,
        })
        
    # ==========================================================================
    # GLOBAL OUT-OF-FOLD EVALUATION & TUNING
    # ==========================================================================
    oof_y_prob = np.concatenate(oof_probs)
    oof_y_true = np.concatenate(oof_labels)
    
    print("\n" + "=" * 65)
    print("  GLOBAL OUT-OF-FOLD EVALUATION (Default Threshold = 0.5)")
    print("=" * 65)
    global_default_metrics = calculate_metrics(oof_y_true, (oof_y_prob >= 0.5).astype(int), oof_y_prob)
    print_evaluation_report(f"{MODEL_NAME} (Global OOF - Default)", global_default_metrics)
    
    print("\n" + "=" * 65)
    print(f"  GLOBAL ACADEMIC INTEGRITY TUNING (FPR <= {TARGET_FPR*100:.1f}%)")
    print("=" * 65)
    global_tuned_threshold, global_tuned_metrics = tune_decision_threshold(oof_y_true, oof_y_prob, target_fpr=TARGET_FPR)
    print_evaluation_report(f"{MODEL_NAME} (Global OOF - Tuned @ {global_tuned_threshold:.3f})", global_tuned_metrics)
    
    # Generate and save plots (confusion matrix & ROC curve)
    from evaluate import plot_global_evaluation
    plot_global_evaluation(oof_y_true, oof_y_prob, global_tuned_threshold, MODEL_NAME, "rf")
    
    # ==========================================================================
    # CROSS-FOLD SUMMARY
    # ==========================================================================
    print("\n" + "=" * 65)
    print(f"  CROSS-FOLD SUMMARY - {MODEL_NAME}")
    print("=" * 65)
    
    metrics_keys = ["accuracy", "f1_score", "auc_roc", "precision",
                    "recall", "specificity", "false_positive_rate"]
    
    print(f"\n{'Metric':<30} {'Mean (default)':>15} {'Mean (tuned)':>14}")
    print("-" * 62)
    for key in metrics_keys:
        default_vals = [s["default"][key] for s in fold_summaries if s["default"][key] is not None]
        tuned_vals   = [s["tuned"][key]   for s in fold_summaries if s["tuned"][key]   is not None]
        mean_def = np.mean(default_vals) if default_vals else float("nan")
        mean_tun = np.mean(tuned_vals)   if tuned_vals   else float("nan")
        label = key.replace("_", " ").title()
        print(f"  {label:<28} {mean_def:>13.4f}  {mean_tun:>13.4f}")
        
    # Save cross-fold summary
    summary_path = os.path.join(OUTPUT_DIR, "rf_cross_fold_summary.json")
    summary_export = []
    for s in fold_summaries:
        summary_export.append({
            "fold"           : s["fold"],
            "best_threshold" : s["best_threshold"],
            "metrics_default": {k: s["default"][k] for k in metrics_keys},
            "metrics_tuned"  : {k: s["tuned"][k]   for k in metrics_keys},
        })
        
    global_results = {
        "model_name": MODEL_NAME,
        "global_default": {k: global_default_metrics[k] for k in metrics_keys},
        "global_tuned": {k: global_tuned_metrics[k] for k in metrics_keys},
        "global_tuned_threshold": global_tuned_threshold,
        "fold_results": summary_export
    }
    
    with open(summary_path, "w") as f:
        json.dump(global_results, f, indent=4)
    print(f"\nCross-fold summary saved -> {summary_path}")
    print("=" * 65)
    print("  Training complete!")
    print("=" * 65)



if __name__ == "__main__":
    run_training()

