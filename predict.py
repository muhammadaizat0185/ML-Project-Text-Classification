# -*- coding: utf-8 -*-
"""
AI Essay Detector - Interactive Prediction Script
Loads pre-trained model pipelines (TF-IDF vectorizers and calibrated classifiers)
and checks whether an input text (copied text or file) is AI-generated or Human-written.
"""

import os
import sys
import joblib
import numpy as np
from scipy.sparse import hstack

# Try importing docx for reading Word files
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

MODELS_DIR = "models"
FOLD = 0 # Default representative fold

def get_docx_text(filepath):
    """Extract paragraphs text from a Word document (.docx)."""
    if not DOCX_AVAILABLE:
        # Fallback to simple zip extraction if docx is not installed
        import zipfile
        import xml.etree.ElementTree as ET
        try:
            with zipfile.ZipFile(filepath) as docx_zip:
                xml_content = docx_zip.read('word/document.xml')
                root = ET.fromstring(xml_content)
                paragraphs = []
                for paragraph in root.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
                    texts = [node.text for node in paragraph.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t') if node.text]
                    paragraphs.append(''.join(texts))
                return '\n'.join(paragraphs)
        except Exception as e:
            raise RuntimeError(f"Failed to read docx via fallback: {e}")
    else:
        doc = docx.Document(filepath)
        return '\n'.join([p.text for p in doc.paragraphs if p.text])

def predict_text(text):
    """Vectorize input text and compute probabilities for SVM and Logistic Regression."""
    fold_dir = os.path.join(MODELS_DIR, f"fold_{FOLD}")
    
    # Check paths
    required_files = [
        "word_vectorizer.joblib",
        "char_vectorizer.joblib",
        "svm_model.joblib",
        "logistic_regression_model.joblib"
    ]
    for f in required_files:
        path = os.path.join(fold_dir, f)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing pre-trained model file: {path}\n"
                "Please make sure the models are trained and saved under the models/ directory."
            )
            
    # Load vectorizers
    word_vectorizer = joblib.load(os.path.join(fold_dir, "word_vectorizer.joblib"))
    char_vectorizer = joblib.load(os.path.join(fold_dir, "char_vectorizer.joblib"))
    
    # Vectorize
    X_word = word_vectorizer.transform([text])
    X_char = char_vectorizer.transform([text])
    X = hstack([X_word, X_char]).tocsr()
    
    # Load classifiers
    svm_model = joblib.load(os.path.join(fold_dir, "svm_model.joblib"))
    lr_model = joblib.load(os.path.join(fold_dir, "logistic_regression_model.joblib"))
    
    # Predict probabilities (Platt scaling / calibrated)
    prob_svm = svm_model.predict_proba(X)[0, 1]
    prob_lr = lr_model.predict_proba(X)[0, 1]
    
    return prob_svm, prob_lr

def print_result_dashboard(text_len, prob_svm, prob_lr):
    """Print a detailed console dashboard of the detection results."""
    # Safety thresholds
    SVM_THRESHOLD = 0.980
    LR_THRESHOLD = 0.817
    
    print("\n" + "=" * 60)
    print("              AI TEXT DETECTION DASHBOARD             ")
    print("=" * 60)
    print(f"  Character length: {text_len} characters")
    print(f"  Approximate words: {len(text_len.split()) if isinstance(text_len, str) else text_len // 6} words")
    print("-" * 60)
    
    # 1. SVM Model
    print("  1. Support Vector Machine (LinearSVC) Model:")
    print(f"     AI Probability Score: {prob_svm * 100:.3f}%")
    if prob_svm >= SVM_THRESHOLD:
        print("     [AI-GENERATED] (HIGH CONFIDENCE)")
        print(f"     Safety Check: Flagged (Exceeds strict safety ceiling of {SVM_THRESHOLD*100:.1f}%)")
    elif prob_svm >= 0.50:
        print("     [AI-GENERATED] (MEDIUM CONFIDENCE / ADVISORY SCREENING)")
        print("     Safety Check: Safe (Does not exceed strict safety ceiling)")
    else:
        print("     [HUMAN-WRITTEN] (LOW AI PROBABILITY)")
        print("     Safety Check: Safe (Low Risk)")
        
    print("-" * 60)
    
    # 2. Logistic Regression Model
    print("  2. Logistic Regression Model:")
    print(f"     AI Probability Score: {prob_lr * 100:.3f}%")
    if prob_lr >= LR_THRESHOLD:
        print("     [AI-GENERATED] (HIGH CONFIDENCE)")
        print(f"     Safety Check: Flagged (Exceeds strict safety ceiling of {LR_THRESHOLD*100:.1f}%)")
    elif prob_lr >= 0.50:
        print("     [AI-GENERATED] (MEDIUM CONFIDENCE / ADVISORY SCREENING)")
        print("     Safety Check: Safe (Does not exceed strict safety ceiling)")
    else:
        print("     [HUMAN-WRITTEN] (LOW AI PROBABILITY)")
        print("     Safety Check: Safe (Low Risk)")
        
    print("=" * 60)

def main():
    print("============================================================")
    print("         Welcome to the Interactive AI Essay Detector       ")
    print("============================================================")
    
    # Check if a file argument is passed in command line
    if len(sys.argv) > 1:
        target_path = sys.argv[1]
    else:
        # Prompt for input
        print("Please choose an input method:")
        print("  1. Check a Word document (.docx) or Text file (.txt)")
        print("  2. Paste text directly in the console")
        choice = input("Enter choice (1 or 2): ").strip()
        
        if choice == "1":
            target_path = input("Enter the file path (e.g. Full Report.docx): ").strip()
        else:
            print("\nPaste your text below (press Enter twice or Ctrl+D/Ctrl+Z then Enter to finish):")
            lines = []
            while True:
                try:
                    line = input()
                    lines.append(line)
                except EOFError:
                    break
            target_path = None
            raw_text = "\n".join(lines)
            
    # Read text
    if target_path:
        # Resolve path quotes if user dragged and dropped file
        target_path = target_path.strip('\'"')
        if not os.path.exists(target_path):
            print(f"Error: File '{target_path}' not found.")
            return
            
        print(f"\nReading text from file: '{target_path}'...")
        try:
            if target_path.endswith(".docx"):
                raw_text = get_docx_text(target_path)
            else:
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
        except Exception as e:
            print(f"Error reading file: {e}")
            return
            
    # Check if text is long enough
    word_count = len(raw_text.split())
    if word_count < 10:
        print("\n[Warning] Input text is too short. AI text detection requires at least 10 words for reliable results.")
        return
        
    print(f"Processed input: {word_count} words.")
    print("Running feature extraction and model inference...")
    
    try:
        prob_svm, prob_lr = predict_text(raw_text)
        print_result_dashboard(len(raw_text), prob_svm, prob_lr)
    except Exception as e:
        import traceback
        print(f"\nInference error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
