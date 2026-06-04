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

---

## 5. Empirical Evaluation Results (Logistic Regression Benchmark)

### A. Fold-by-Fold Performance (Default Threshold = 0.5)
The dataset was split into 5 stratified folds. The model shows extremely consistent performance across all folds, indicating high generalization stability:
* **Fold 0**: Accuracy: **98.31%** | F1-Score: **0.9864** | ROC-AUC: **0.9982** | FPR: **0.840%**
* **Fold 1**: Accuracy: **98.12%** | F1-Score: **0.9848** | ROC-AUC: **0.9975** | FPR: **1.060%**
* **Fold 2**: Accuracy: **98.08%** | F1-Score: **0.9845** | ROC-AUC: **0.9977** | FPR: **1.352%**
* **Fold 3**: Accuracy: **98.21%** | F1-Score: **0.9855** | ROC-AUC: **0.9980** | FPR: **1.060%**
* **Fold 4**: Accuracy: **98.12%** | F1-Score: **0.9848** | ROC-AUC: **0.9976** | FPR: **1.188%**

---

### B. Global Out-of-Fold (OOF) Comparison
Below is the comparison of the global OOF metrics on all 72,991 essays under the default decision threshold and the optimized academic integrity threshold.

| Metric | Default Threshold (0.5) | Tuned Threshold (0.817) | Impact / Rationale |
| :--- | :--- | :--- | :--- |
| **Decision Threshold** | `0.500` | `0.817` | Threshold is shifted higher to protect students. |
| **Accuracy** | **98.166%** | **96.195%** | Minor drop in accuracy to ensure safety. |
| **F1-Score** | **0.9852** | **0.9686** | High balanced score maintained. |
| **ROC-AUC** | **0.9978** | **0.9978** | Unchanged (reflects model discrimination capacity). |
| **Precision** | **99.329%** | **99.937%** | Higher precision means almost zero false flags. |
| **Recall (Detection)** | **97.725%** | **93.973%** | Detection remains high; catches ~94% of AI text. |
| **Specificity** | **98.900%** | **99.901%** | Human essays correctly classified increases to 99.9%. |
| **False Positive Rate** | **1.100%** *(HIGH RISK)* | **0.099%** *(SAFE)* | **FPR reduced below target of 0.1%**. |
| **False Accusations (Count)**| **301 essays** | **27 essays** | **FPR reduced by 91.03% (274 fewer false accusations)**. |

---

### C. Confusion Matrices

#### 1. Default Threshold (0.5)
```text
                  Predicted Human  |  Predicted AI
Actual Human          27,062       |      301       <-- Falsely Accused
Actual AI              1,038       |   44,590       <-- Detected AI
```

#### 2. Tuned Threshold (0.817)
```text
                  Predicted Human  |  Predicted AI
Actual Human          27,336       |       27       <-- Falsely Accused (91% reduction!)
Actual AI              2,750       |   42,878       <-- Detected AI
```

