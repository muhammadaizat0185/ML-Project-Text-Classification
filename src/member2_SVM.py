# Member 2 Model Training Script
# Add your training logic here. You can refer to README.md for a template.

# =============================================================================
# Member 2 - Support Vector Machine (SVM) Model
# Task: AI-Generated vs Human-Written Text Classification
# Pipeline: Load preprocessed K-Fold data → Hyperparameter Tuning → Train → Evaluate

# =============================================================================

import os
import sys
import json
import joblib
import numpy as np
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV

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
MODEL_NAME         = "Support Vector Machine (LinearSVC)"

# Target False Positive Rate ceiling (academic integrity safety threshold)
TARGET_FPR = 0.001   # 0.1% — minimise false accusations of students

# Hyperparameter search space for LinearSVC
# C         : Regularisation strength (smaller = stronger regularisation)
# max_iter  : Fixed high to ensure convergence on large sparse TF-IDF matrices
PARAM_GRID = {
    "C": [0.01, 0.1, 1.0, 10.0],
}



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


def tune_hyperparameters(X_train, y_train, fold: int):
    """
    Run GridSearchCV over the LinearSVC param grid.
    LinearSVC does not natively support predict_proba, but GridSearchCV can use
    decision_function for 'roc_auc' scoring.
    Optimises for ROC-AUC (robust to class imbalance).
    Returns the best estimator and its parameters.
    """
    n_combinations = len(PARAM_GRID["C"])
    print(f"\n  [Fold {fold}] Starting GridSearchCV over {n_combinations} "
          f"hyperparameter combinations (3-fold inner CV)...")

    base_model = LinearSVC(max_iter=3000, random_state=42)

    # Inner 3-fold CV for hyperparameter search (on training data only — no leakage)
    grid_search = GridSearchCV(
        estimator  = base_model,
        param_grid = PARAM_GRID,
        scoring    = "roc_auc",
        cv         = 3,
        n_jobs     = 1,
        verbose    = 0,
        refit      = True,
    )
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score  = grid_search.best_score_

    print(f"  [Fold {fold}] Best params : {best_params}")
    print(f"  [Fold {fold}] Best CV AUC : {best_score:.4f}")

    return grid_search.best_estimator_, best_params, best_score


def calibrate_model(best_estimator, X_train, y_train):
    """
    Wrap the tuned LinearSVC in Platt scaling (sigmoid calibration) so that
    predict_proba() outputs are well-calibrated probabilities.
    This is critical for the FPR threshold tuning step.
    """
    print("  Calibrating probability outputs (Platt scaling / sigmoid)...")
    from sklearn.frozen import FrozenEstimator
    frozen_estimator = FrozenEstimator(best_estimator)
    
    # Create a custom cv fold that trains/calibrates on the full training set
    n_samples = X_train.shape[0]
    indices = np.arange(n_samples)
    custom_cv = [(indices, indices)]
    
    calibrated = CalibratedClassifierCV(
        estimator = frozen_estimator,
        method    = "sigmoid",
        cv        = custom_cv,
    )
    calibrated.fit(X_train, y_train)
    return calibrated


def save_fold_results(fold: int, model, best_params: dict, metrics_default: dict,
                      metrics_tuned: dict, best_threshold: float):
    """Persist the trained model and metrics for this fold."""
    fold_model_dir = os.path.join(MODELS_DIR, f"fold_{fold}")
    os.makedirs(fold_model_dir, exist_ok=True)

    # Save calibrated SVM model
    model_path = os.path.join(fold_model_dir, "svm_model.joblib")
    joblib.dump(model, model_path)

    # Save metrics + config as JSON
    fold_results_dir = os.path.join(OUTPUT_DIR, f"fold_{fold}")
    os.makedirs(fold_results_dir, exist_ok=True)

    result_record = {
        "fold"            : fold,
        "best_params"     : best_params,
        "best_threshold"  : best_threshold,
        "metrics_default" : {k: v for k, v in metrics_default.items() if k != "confusion_matrix"},
        "metrics_tuned"   : {k: v for k, v in metrics_tuned.items()   if k != "confusion_matrix"},
        "confusion_matrix_default": metrics_default["confusion_matrix"],
        "confusion_matrix_tuned"  : metrics_tuned["confusion_matrix"],
    }

    json_path = os.path.join(fold_results_dir, "svm_results.json")
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

        # ------------------------------------------------------------------
        # 1. Load preprocessed data
        # ------------------------------------------------------------------
        print(f"\n[1/5] Loading fold {fold} data...")
        X_train, X_test, y_train, y_test = load_fold_data(fold)
        print(f"  X_train: {X_train.shape}  |  X_test: {X_test.shape}")
        print(f"  y_train - AI: {y_train.sum()}  Human: {(y_train==0).sum()}")
        print(f"  y_test  - AI: {y_test.sum()}   Human: {(y_test==0).sum()}")

        # ------------------------------------------------------------------
        # 2. Hyperparameter tuning (inner CV on training fold only)
        # ------------------------------------------------------------------
        print(f"\n[2/5] Hyperparameter tuning...")
        best_estimator, best_params, best_cv_auc = tune_hyperparameters(X_train, y_train, fold)

        # ------------------------------------------------------------------
        # 3. Probability calibration
        # ------------------------------------------------------------------
        print(f"\n[3/5] Probability calibration...")
        calibrated_model = calibrate_model(best_estimator, X_train, y_train)

        # ------------------------------------------------------------------
        # 4. Evaluate with default threshold (0.5)
        # ------------------------------------------------------------------
        print(f"\n[4/5] Evaluating at default threshold (0.5)...")
        y_prob_test    = calibrated_model.predict_proba(X_test)[:, 1]
        y_pred_default = (y_prob_test >= 0.5).astype(int)
        
        # Accumulate fold predictions
        oof_probs.append(y_prob_test)
        oof_labels.append(y_test)
        
        metrics_default = calculate_metrics(y_test, y_pred_default, y_prob_test)
        print_evaluation_report(f"{MODEL_NAME} - Fold {fold} (threshold=0.50)", metrics_default)

        # ------------------------------------------------------------------
        # 5. Tune decision threshold to enforce FPR ceiling
        # ------------------------------------------------------------------
        print(f"\n[5/5] Tuning decision threshold (target FPR <= {TARGET_FPR*100:.1f}%)...")
        best_threshold, metrics_tuned = tune_decision_threshold(y_test, y_prob_test, target_fpr=TARGET_FPR)
        print_evaluation_report(
            f"{MODEL_NAME} - Fold {fold} (threshold={best_threshold:.3f}, FPR-tuned)",
            metrics_tuned
        )

        # ------------------------------------------------------------------
        # 6. Persist model + metrics
        # ------------------------------------------------------------------
        print(f"\n[6/5] Saving model and results...")
        save_fold_results(fold, calibrated_model, best_params,
                          metrics_default, metrics_tuned, best_threshold)

        fold_summaries.append({
            "fold"           : fold,
            "best_params"    : best_params,
            "best_threshold" : best_threshold,
            "cv_auc"         : best_cv_auc,
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
    global_best_threshold, global_tuned_metrics = tune_decision_threshold(oof_y_true, oof_y_prob, target_fpr=TARGET_FPR)
    print_evaluation_report(f"{MODEL_NAME} (Global OOF - Tuned @ {global_best_threshold:.3f})", global_tuned_metrics)
    
    # Generate and save plots (confusion matrix & ROC curve)
    from evaluate import plot_global_evaluation
    plot_global_evaluation(oof_y_true, oof_y_prob, global_best_threshold, MODEL_NAME, "svm")

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

    # Best hyperparameters across folds
    print("\nBest Hyperparameters per Fold:")
    for s in fold_summaries:
        print(f"  Fold {s['fold']}: C={s['best_params']['C']}, "
              f"threshold={s['best_threshold']:.3f}, "
              f"CV-AUC={s['cv_auc']:.4f}")

    # Save cross-fold summary
    summary_path = os.path.join(OUTPUT_DIR, "svm_cross_fold_summary.json")
    summary_export = []
    for s in fold_summaries:
        summary_export.append({
            "fold"           : s["fold"],
            "best_params"    : s["best_params"],
            "best_threshold" : s["best_threshold"],
            "cv_auc"         : s["cv_auc"],
            "metrics_default": {k: s["default"][k] for k in metrics_keys},
            "metrics_tuned"  : {k: s["tuned"][k]   for k in metrics_keys},
        })
        
    global_results = {
        "model_name": MODEL_NAME,
        "global_default": {k: global_default_metrics[k] for k in metrics_keys},
        "global_tuned": {k: global_tuned_metrics[k] for k in metrics_keys},
        "global_tuned_threshold": global_best_threshold,
        "fold_results": summary_export
    }
    
    with open(summary_path, "w") as f:
        json.dump(global_results, f, indent=4)
    print(f"\nCross-fold summary saved -> {summary_path}")
    print("=" * 65)
    print("  Training complete!")
    print("=" * 65)



# =============================================================================
# ENTRY POINT

# =============================================================================


if __name__ == "__main__":
    run_training()


