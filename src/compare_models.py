# -*- coding: utf-8 -*-
"""
Model Comparison & Visualization Script
Loads the cross-fold results of Logistic Regression, SVM, Naive Bayes, and Random Forest,
prints a markdown-style comparison table, and saves comparative visualization plots.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
IMAGE_DIR = "results"  # Output directory for images

# Configure plotting style
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16
})

def load_results():
    """Load cross-fold summaries from JSON files."""
    models_data = {}
    
    file_mapping = {
        "Logistic Regression": "logistic_cross_fold_summary.json",
        "Support Vector Machine": "svm_cross_fold_summary.json",
        "Multinomial Naive Bayes": "mnb_cross_fold_summary.json",
        "Random Forest": "rf_cross_fold_summary.json"
    }
    
    for name, filename in file_mapping.items():
        path = os.path.join(RESULTS_DIR, filename)
        if os.path.exists(path):
            with open(path, "r") as f:
                models_data[name] = json.load(f)
        else:
            print(f"Warning: Summary file {filename} not found at {path}.")
            
    return models_data

def compile_comparison_dataframe(models_data):
    """Transform the loaded JSON results into a clean pandas DataFrame for comparisons."""
    records = []
    metrics_keys = ["accuracy", "f1_score", "auc_roc", "precision", "recall", "specificity", "false_positive_rate"]
    
    for model_name, data in models_data.items():
        # Robust schema parsing (list of folds vs dict wrapper)
        if isinstance(data, list):
            fold_results = data
            thresholds = [f["best_threshold"] for f in fold_results if "best_threshold" in f]
            avg_threshold = np.mean(thresholds) if thresholds else 0.5
        elif isinstance(data, dict):
            fold_results = data["fold_results"]
            avg_threshold = data.get("global_tuned_threshold", 0.5)
        else:
            print(f"Error: Format of {model_name} results is unrecognized.")
            continue
            
        # Calculate mean metrics across all folds
        def_metrics = {}
        for key in metrics_keys:
            vals = [f["metrics_default"][key] for f in fold_results if key in f["metrics_default"]]
            def_metrics[key] = np.mean(vals) if vals else 0.0
            
        tuned_metrics = {}
        for key in metrics_keys:
            vals = [f["metrics_tuned"][key] for f in fold_results if key in f["metrics_tuned"]]
            tuned_metrics[key] = np.mean(vals) if vals else 0.0
            
        # 1. Default Threshold (0.5)
        rec_default = {
            "Model": model_name,
            "Setting": "Default (0.50)",
            "Accuracy": def_metrics["accuracy"] * 100,
            "F1-Score": def_metrics["f1_score"],
            "ROC-AUC": def_metrics["auc_roc"],
            "Precision": def_metrics["precision"] * 100,
            "Recall (Detection)": def_metrics["recall"] * 100,
            "Specificity": def_metrics["specificity"] * 100,
            "FPR": def_metrics["false_positive_rate"] * 100
        }
        records.append(rec_default)
        
        # 2. Tuned Safety Threshold
        rec_tuned = {
            "Model": model_name,
            "Setting": f"Tuned ({avg_threshold:.3f})",
            "Accuracy": tuned_metrics["accuracy"] * 100,
            "F1-Score": tuned_metrics["f1_score"],
            "ROC-AUC": tuned_metrics["auc_roc"],
            "Precision": tuned_metrics["precision"] * 100,
            "Recall (Detection)": tuned_metrics["recall"] * 100,
            "Specificity": tuned_metrics["specificity"] * 100,
            "FPR": tuned_metrics["false_positive_rate"] * 100
        }
        records.append(rec_tuned)
        
    return pd.DataFrame(records)

def plot_roc_auc_comparison(df_comparison):
    """Generate and save a bar chart comparing the ROC-AUC of all models."""
    plt.figure(figsize=(8, 5))
    
    # Filter default settings (AUC is the same for default/tuned, so we just pick one)
    df_auc = df_comparison[df_comparison["Setting"].str.contains("Default")].copy()
    df_auc = df_auc.sort_values(by="ROC-AUC", ascending=False)
    
    colors = sns.color_palette("viridis", len(df_auc))
    bars = plt.bar(df_auc["Model"], df_auc["ROC-AUC"], color=colors, edgecolor='grey', width=0.5)
    
    # Highlight value labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontweight='bold')
        
    plt.ylim(0.90, 1.01)  # Focus on the high-performance range
    plt.ylabel("ROC-AUC Score")
    plt.title("Classifier Discrimination Capability (ROC-AUC)")
    plt.tight_layout()
    
    plot_path = os.path.join(IMAGE_DIR, "model_comparison_roc_auc.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved ROC-AUC comparison plot -> {plot_path}")

def plot_safety_recall_comparison(df_comparison):
    """Generate and save a chart comparing the Recall (AI detection) under tuned safety threshold."""
    plt.figure(figsize=(9, 5.5))
    
    # Filter tuned safety settings
    df_tuned = df_comparison[df_comparison["Setting"].str.contains("Tuned")].copy()
    df_tuned = df_tuned.sort_values(by="Recall (Detection)", ascending=False)
    
    colors = sns.color_palette("magma", len(df_tuned))
    bars = plt.bar(df_tuned["Model"], df_tuned["Recall (Detection)"], color=colors, edgecolor='grey', width=0.5)
    
    # Add values on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontweight='bold')
        
    plt.ylim(0, 110)
    plt.ylabel("Recall / AI Detection Rate (%)")
    plt.xlabel("Model")
    plt.title("Recall / Detection Rate Under Strict Academic Integrity safety (FPR <= 0.1%)")
    plt.tight_layout()
    
    plot_path = os.path.join(IMAGE_DIR, "model_comparison_tuned_recall.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved Tuned Recall comparison plot -> {plot_path}")

def plot_fpr_default_vs_tuned(df_comparison):
    """Plot comparative FPR rates before and after safety threshold tuning."""
    plt.figure(figsize=(9, 5.5))
    
    # Restructure dataframe for grouped bar plotting
    df_plot = df_comparison.copy()
    df_plot["Setting Type"] = df_plot["Setting"].apply(lambda x: "Default (0.50)" if "Default" in x else "Tuned (FPR <= 0.1%)")
    
    ax = sns.barplot(
        data=df_plot,
        x="Model",
        y="FPR",
        hue="Setting Type",
        palette="muted"
    )
    
    # Add value annotations on top of the bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f"{height:.3f}%",
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center',
                        xytext=(0, 7),
                        textcoords='offset points',
                        fontweight='bold', fontsize=9)
            
    # Add red line for target FPR boundary
    plt.axhline(0.1, color='red', linestyle='--', linewidth=1.5, label="Safety Threshold Ceiling (0.1%)")
    
    plt.ylabel("False Positive Rate (%)")
    plt.title("False Positive Rate (False Accusations): Default vs Tuned Threshold")
    plt.legend(loc="upper right")
    plt.tight_layout()
    
    plot_path = os.path.join(IMAGE_DIR, "model_comparison_fpr_reduction.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved FPR Comparison plot -> {plot_path}")

def main():
    print("=== Loading Model Summaries ===")
    models_data = load_results()
    
    if not models_data:
        print("No model results available. Please run the training scripts first.")
        return
        
    df_comparison = compile_comparison_dataframe(models_data)
    
    print("\n=== GLOBAL MODEL COMPARISON TABLE (CROSS-FOLD MEANS) ===")
    cols_to_show = ["Model", "Setting", "Accuracy", "F1-Score", "ROC-AUC", "Precision", "Recall (Detection)", "FPR"]
    print(df_comparison[cols_to_show].to_string(index=False))
    
    print("\n=== Generating Comparative Visualization Plots ===")
    plot_roc_auc_comparison(df_comparison)
    plot_safety_recall_comparison(df_comparison)
    plot_fpr_default_vs_tuned(df_comparison)
    print("\nVisualizations saved successfully to the results/ folder.")


if __name__ == "__main__":
    main()


