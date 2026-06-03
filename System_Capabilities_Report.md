# 📊 Project Capabilities & System Architecture Report

This report outlines the technical capabilities, architecture, and current execution pipeline of the **Generative AI Text Detection System**.

---

## 1. Executive Summary
The system is built to classify academic essays as either **Human-Written** or **AI-Generated** using a high-dimensional text classification pipeline. To ensure maximum reliability and statistical rigor, the project is structured around a **5-Fold Stratified Cross-Validation** design. The system is specifically optimized for **Academic Integrity Safety** by enforcing a decision boundary that limits the False Positive Rate (FPR) to $\le 0.1\%$ to protect students from false accusations.

---

## 2. Core System Capabilities

### 🔍 A. Quality Control & Data Cleaning (`src/clean_dataset.py`)
* **Noise Filtering**: Identifies and removes corrupted/truncated inputs (e.g., short prompt titles, 2-to-3-word snippets, and gibberish). A minimum threshold of **100 words** is enforced to ensure only complete essay bodies are analyzed.
* **Whitespace Normalization**: Standardizes character spacings and strips trailing/leading whitespaces from text bodies.
* **Corpus Integrity**: Resolves data quality issues without modifying the label distributions.
* **Status**: Completed. Run on the original dataset, resulting in:
  * **Original Dataset**: 73,573 rows
  * **Cleaned Dataset**: 72,991 rows (582 corrupted rows filtered)
  * **Class Balance**: 62.51% AI-Generated (1), 37.49% Human-Written (0)

### 📈 B. Stratified 5-Fold Splitting (`src/preprocess.py`)
* **Leakage Prevention**: Instead of a single train-test split, the corpus is split into 5 distinct folds using `StratifiedKFold`. Preprocessing vectorizers are fit **only** on training folds and then applied to testing folds. This ensures zero data leakage.
* **Ratio**: Each fold sets up an 80/20 train/test partition, representing the entire corpus across the 5 iterations.
* **Directory Structure**: Automatically outputs split indices and feature matrices to separate fold folders (`processed_data/fold_i/`), making multi-member parallel coding easy.

### 🔠 C. Dual TF-IDF Feature Extraction (`src/preprocess.py`)
* **Step 1: Tokenization and Case Normalization**: Standardizes vocabulary by converting all words to lowercase and parsing strings into tokens.
* **Step 2: Stop-Word Elimination & Noise Reduction**: Filters out highly frequent, non-informative English structural words (e.g., "the", "is", "at") to highlight core vocabulary and content indicators.
* **Step 3: Hybrid High-Dimensional Vectorization**:
  * **Word-level TF-IDF**: Extracts 1-to-2 n-grams (up to 25,000 features) to analyze structural transitions and vocabulary choices.
  * **Character-level TF-IDF**: Extracts 3-to-5 char n-grams (up to 25,000 features) to identify spelling anomalies, sub-word tokens, punctuation density, and stylistic patterns.
  * Both matrices are concatenated into a single sparse matrix of **50,000 features per row**.

### 🛠️ D. Multi-Model Collaborative Architecture (`src/`)
* **Model 1: Logistic Regression** ([src/member1_logistic.py](file:///c:/Users/muham/Documents/Machine%20Learning/Project/src/member1_logistic.py)): Handled by Member 1. Performs hyperparameter tuning (regularization strength `C`) using nested Stratified 3-Fold Grid Search on each training fold.
* **Model 2: Support Vector Machine (SVM)** ([src/member2_SVM.py](file:///c:/Users/muham/Documents/Machine%20Learning/Project/src/member2_SVM.py)): Handled by Member 2. Handled via template files.
* **Model 3: Multinomial Naive Bayes** ([src/member3_Multinomial_NaiveBayes.py](file:///c:/Users/muham/Documents/Machine%20Learning/Project/src/member3_Multinomial_NaiveBayes.py)): Handled by Member 3. Handled via template files.
* **Model 4: Random Forest** ([src/member4_RandomForest.py](file:///c:/Users/muham/Documents/Machine%20Learning/Project/src/member4_RandomForest.py)): Handled by Member 4. Handled via template files.

### 🛡️ E. Academic Integrity Guardrails (`src/evaluate.py`)
* **Threshold Optimization**: Incorporates a scanning algorithm to shift decision thresholds from the default `0.5`.
* **Strict FPR Boundary**: Iterates through 1,000 potential probability cut-offs to lock down the exact threshold where the **False Positive Rate is $\le 0.1\%$** (maximum 1 false accusation per 1,000 human essays) while maximizing detection rates.

---

## 3. Current Implementation Status

| Component | Status | File / Folder |
| :--- | :--- | :--- |
| Data Cleaning | **Complete** | [clean_dataset.py](file:///c:/Users/muham/Documents/Machine%20Learning/Project/src/clean_dataset.py) |
| Feature Extraction & 5-Fold Split | **In Progress** | [preprocess.py](file:///c:/Users/muham/Documents/Machine%20Learning/Project/src/preprocess.py) |
| Logistic Regression Pipeline | **Complete** | [member1_logistic.py](file:///c:/Users/muham/Documents/Machine%20Learning/Project/src/member1_logistic.py) |
| SVM Codebase Shell | **Created** | [member2_SVM.py](file:///c:/Users/muham/Documents/Machine%20Learning/Project/src/member2_SVM.py) |
| Naive Bayes Codebase Shell | **Created** | [member3_Multinomial_NaiveBayes.py](file:///c:/Users/muham/Documents/Machine%20Learning/Project/src/member3_Multinomial_NaiveBayes.py) |
| Random Forest Codebase Shell | **Created** | [member4_RandomForest.py](file:///c:/Users/muham/Documents/Machine%20Learning/Project/src/member4_RandomForest.py) |
| Evaluation & Threshold Tuner | **Complete** | [evaluate.py](file:///c:/Users/muham/Documents/Machine%20Learning/Project/src/evaluate.py) |
| GitHub Repository Integration | **Active** | [Link to Repo](https://github.com/muhammadaizat0185/ML-Project-Text-Classification) |

---

## 4. How to Run the Pipeline

1. **Clean Data**:
   ```bash
   python src/clean_dataset.py
   ```
2. **Preprocess and Extract Features**:
   ```bash
   python src/preprocess.py
   ```
3. **Train and Evaluate Logistic Regression**:
   ```bash
   python src/member1_logistic.py
   ```
