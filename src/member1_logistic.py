import os
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from evaluate import calculate_metrics, print_evaluation_report, tune_decision_threshold

def train_logistic_regression(data_dir, models_dir):
    print("=== Member 1: Loading Shared Preprocessed Folds ===")
    
    n_folds = 5
    oof_probs = []
    oof_labels = []
    fold_metrics_list = []
    
    print(f"Starting 5-Fold Cross-Validation training...")
    
    for fold in range(n_folds):
        print(f"\n========================================")
        print(f" TRAINING FOLD {fold}")
        print(f"========================================")
        fold_data_dir = os.path.join(data_dir, f"fold_{fold}")
        fold_models_dir = os.path.join(models_dir, f"fold_{fold}")
        
        X_train_path = os.path.join(fold_data_dir, "X_train.npz")
        X_test_path = os.path.join(fold_data_dir, "X_test.npz")
        y_train_path = os.path.join(fold_data_dir, "y_train.npy")
        y_test_path = os.path.join(fold_data_dir, "y_test.npy")
        
        if not (os.path.exists(X_train_path) and os.path.exists(X_test_path)):
            raise FileNotFoundError(f"Preprocessed sparse matrices for Fold {fold} not found. Please run preprocess.py first.")
            
        X_train = load_npz(X_train_path)
        X_test = load_npz(X_test_path)
        y_train = np.load(y_train_path)
        y_test = np.load(y_test_path)
        
        print(f"X_train Shape: {X_train.shape}")
        print(f"X_test Shape:  {X_test.shape}")
        print(f"y_train distribution: AI (1)={np.sum(y_train)}, Human (0)={len(y_train) - np.sum(y_train)}")
        
        # Define hyperparameter grid for Logistic Regression
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l2']
        }
        
        base_model = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)
        
        # 3-Fold CV on the training fold (nested CV)
        print(f"Running GridSearchCV for Fold {fold} with 3-fold cross-validation...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            scoring='roc_auc',
            cv=3,
            n_jobs=1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        print(f"Best Hyperparameters for Fold {fold}: {grid_search.best_params_}")
        print(f"Best CV ROC-AUC for Fold {fold}:            {grid_search.best_score_:.4f}")
        
        # Predict default probabilities and classes
        y_prob = best_model.predict_proba(X_test)[:, 1]
        y_pred_default = (y_prob >= 0.5).astype(int)
        
        oof_probs.append(y_prob)
        oof_labels.append(y_test)
        
        # Calculate individual fold metrics (at default threshold 0.5)
        fold_metrics = calculate_metrics(y_test, y_pred_default, y_prob)
        fold_metrics_list.append(fold_metrics)
        
        # Save model for this fold
        os.makedirs(fold_models_dir, exist_ok=True)
        model_save_path = os.path.join(fold_models_dir, "logistic_regression_model.joblib")
        joblib.dump(best_model, model_save_path)
        print(f"Fold {fold} Model saved successfully to: {model_save_path}")
        
    # Concatenate Out-Of-Fold (OOF) arrays
    oof_y_prob = np.concatenate(oof_probs)
    oof_y_true = np.concatenate(oof_labels)
    
    print("\n========================================")
    print(" FOLD-BY-FOLD SUMMARY (Default Threshold = 0.5)")
    print("========================================")
    for fold, fm in enumerate(fold_metrics_list):
        print(f"Fold {fold} | Accuracy: {fm['accuracy']*100:.2f}% | F1: {fm['f1_score']:.4f} | ROC-AUC: {fm['auc_roc']:.4f} | FPR: {fm['false_positive_rate']*100:.3f}%")
        
    print("\n========================================")
    print(" PHASE 2: GLOBAL OUT-OF-FOLD EVALUATION (Default Threshold = 0.5)")
    print("========================================")
    oof_default_metrics = calculate_metrics(oof_y_true, (oof_y_prob >= 0.5).astype(int), oof_y_prob)
    print_evaluation_report("Logistic Regression (Global OOF - Default)", oof_default_metrics)
    
    print("\n========================================")
    print(" PHASE 3: GLOBAL ACADEMIC INTEGRITY TUNING (FPR <= 0.1%)")
    print("========================================")
    target_fpr = 0.001
    best_threshold, tuned_metrics = tune_decision_threshold(oof_y_true, oof_y_prob, target_fpr=target_fpr)
    print_evaluation_report(f"Logistic Regression (Global OOF - Tuned Threshold @ {best_threshold:.3f})", tuned_metrics)
    
    # Generate and save comparative plots (confusion matrix & ROC curve)
    from evaluate import plot_global_evaluation
    plot_global_evaluation(oof_y_true, oof_y_prob, best_threshold, "Logistic Regression", "logistic")
    
    # Save results to json summary matching the format of other models
    OUTPUT_DIR = "results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    summary_path = os.path.join(OUTPUT_DIR, "logistic_cross_fold_summary.json")
    
    metrics_keys = ["accuracy", "f1_score", "auc_roc", "precision",
                    "recall", "specificity", "false_positive_rate"]
    
    summary_export = []
    for fold, fm in enumerate(fold_metrics_list):
        y_prob_fold = oof_probs[fold]
        y_test_fold = oof_labels[fold]
        tuned_fm = calculate_metrics(y_test_fold, (y_prob_fold >= best_threshold).astype(int), y_prob_fold)
        summary_export.append({
            "fold": fold,
            "best_threshold": best_threshold,
            "cv_auc": fm["auc_roc"],
            "metrics_default": {k: fm[k] for k in metrics_keys},
            "metrics_tuned": {k: tuned_fm[k] for k in metrics_keys},
        })
        
    global_results = {
        "model_name": "Logistic Regression",
        "global_default": {k: oof_default_metrics[k] for k in metrics_keys},
        "global_tuned": {k: tuned_metrics[k] for k in metrics_keys},
        "global_tuned_threshold": best_threshold,
        "fold_results": summary_export
    }
    
    import json
    with open(summary_path, "w") as f:
        json.dump(global_results, f, indent=4)
    print(f"\nCross-fold summary saved -> {summary_path}")



if __name__ == "__main__":
    DATA_DIR = "processed_data"
    MODELS_DIR = "models"
    
    train_logistic_regression(DATA_DIR, MODELS_DIR)


