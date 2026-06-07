# -*- coding: utf-8 -*-
"""
Generate a detailed model performance dashboard.
Compares Accuracy, F1-score, Precision, Recall, and Specificity across Default and Tuned configurations.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = "results"
IMAGE_PATH = os.path.join(RESULTS_DIR, "model_comparison_detailed_dashboard.png")

def load_results():
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
            print(f"Warning: Summary file {filename} not found.")
    return models_data

def prepare_data(models_data):
    records = []
    metrics = ["accuracy", "f1_score", "precision", "recall", "specificity", "false_positive_rate"]
    
    for model_name, data in models_data.items():
        if isinstance(data, list):
            fold_results = data
        elif isinstance(data, dict):
            fold_results = data["fold_results"]
        else:
            continue
            
        for config_type in ["default", "tuned"]:
            key_metrics = {}
            for m in metrics:
                vals = [f[f"metrics_{config_type}"][m] for f in fold_results if m in f[f"metrics_{config_type}"]]
                key_metrics[m] = np.mean(vals) if vals else 0.0
                
            records.append({
                "Model": model_name,
                "Configuration": "Default (Threshold=0.5)" if config_type == "default" else "Tuned (Safety)",
                "Accuracy": key_metrics["accuracy"] * 100,
                "F1-Score": key_metrics["f1_score"] * 100,
                "Precision": key_metrics["precision"] * 100,
                "Recall (Detection)": key_metrics["recall"] * 100,
                "Specificity": key_metrics["specificity"] * 100,
                "FPR": key_metrics["false_positive_rate"] * 100
            })
            
    return pd.DataFrame(records)

def build_dashboard():
    models_data = load_results()
    if not models_data:
        print("No results loaded.")
        return
        
    df = prepare_data(models_data)
    
    # Set up subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    fig.suptitle("AI Essay Detector - Comprehensive Classifier Performance Analysis", fontsize=16, fontweight='bold', y=0.98)
    
    # Subplot 1: Accuracy Comparison
    sns.barplot(data=df, x="Model", y="Accuracy", hue="Configuration", ax=axes[0, 0], palette="Blues_r")
    axes[0, 0].set_title("Accuracy Comparison (Default vs Tuned)", fontweight='bold')
    axes[0, 0].set_ylabel("Accuracy (%)")
    axes[0, 0].set_ylim(0, 110)
    for p in axes[0, 0].patches:
        h = p.get_height()
        if h > 0:
            axes[0, 0].annotate(f"{h:.1f}%", (p.get_x() + p.get_width()/2., h), ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    # Subplot 2: F1-Score Comparison
    sns.barplot(data=df, x="Model", y="F1-Score", hue="Configuration", ax=axes[0, 1], palette="Purples_r")
    axes[0, 1].set_title("F1-Score Comparison (Default vs Tuned)", fontweight='bold')
    axes[0, 1].set_ylabel("F1-Score (%)")
    axes[0, 1].set_ylim(0, 110)
    for p in axes[0, 1].patches:
        h = p.get_height()
        if h > 0:
            axes[0, 1].annotate(f"{h:.1f}%", (p.get_x() + p.get_width()/2., h), ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    # Subplot 3: Recall (AI Detection Rate)
    sns.barplot(data=df, x="Model", y="Recall (Detection)", hue="Configuration", ax=axes[1, 0], palette="Oranges_r")
    axes[1, 0].set_title("Recall / Detection Sensitivity (Default vs Tuned)", fontweight='bold')
    axes[1, 0].set_ylabel("Recall (%)")
    axes[1, 0].set_ylim(0, 110)
    for p in axes[1, 0].patches:
        h = p.get_height()
        if h > 0:
            axes[1, 0].annotate(f"{h:.1f}%", (p.get_x() + p.get_width()/2., h), ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    # Subplot 4: Specificity (Academic Safety Rate)
    sns.barplot(data=df, x="Model", y="Specificity", hue="Configuration", ax=axes[1, 1], palette="Greens_r")
    axes[1, 1].set_title("Specificity / Safe Student Classification (Default vs Tuned)", fontweight='bold')
    axes[1, 1].set_ylabel("Specificity (%)")
    axes[1, 1].set_ylim(0, 110)
    for p in axes[1, 1].patches:
        h = p.get_height()
        if h > 0:
            axes[1, 1].annotate(f"{h:.1f}%", (p.get_x() + p.get_width()/2., h), ha='center', va='bottom', fontsize=9, fontweight='bold')
            
    # Layout adjustments
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(IMAGE_PATH, bbox_inches='tight')
    plt.close()
    print(f"Detailed performance dashboard saved successfully to {IMAGE_PATH}")

if __name__ == "__main__":
    build_dashboard()
