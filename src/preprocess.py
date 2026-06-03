import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, save_npz

def run_preprocessing(input_csv, output_dir, models_dir):
    print("=== Loading Cleaned Dataset ===")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Cleaned dataset not found at {input_csv}. Please run clean_dataset.py first.")
        
    df = pd.read_csv(input_csv)
    print(f"Loaded {df.shape[0]} rows.")
    
    # Fill any NaNs just in case
    df['text'] = df['text'].fillna('')
    
    print("\n=== Performing 5-Fold Stratified Split ===")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n--- Processing Fold {fold} ---")
        train_df = df.iloc[train_idx].copy()
        test_df = df.iloc[test_idx].copy()
        
        print(f"Training set size: {train_df.shape[0]} rows")
        print(f"Testing set size: {test_df.shape[0]} rows")
        
        # Track label distributions
        for dataset_name, subset in [("Train", train_df), ("Test", test_df)]:
            counts = subset['label'].value_counts()
            print(f"  {dataset_name} set - AI (1): {counts.get(1, 0)} ({counts.get(1, 0)/len(subset)*100:.2f}%), "
                  f"Human (0): {counts.get(0, 0)} ({counts.get(0, 0)/len(subset)*100:.2f}%)")
                  
        print("Fitting Word-level TF-IDF (n-grams: 1 to 2, max features: 25,000, stop words: english)...")
        word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=25000,
            sublinear_tf=True,
            stop_words='english'
        )
        
        # Fit only on the training set to prevent data leakage!
        X_train_word = word_vectorizer.fit_transform(train_df['text'])
        X_test_word = word_vectorizer.transform(test_df['text'])
        
        print("Fitting Character-level TF-IDF (char n-grams: 3 to 5, max features: 25,000)...")
        char_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            max_features=25000,
            sublinear_tf=True
        )
        
        X_train_char = char_vectorizer.fit_transform(train_df['text'])
        X_test_char = char_vectorizer.transform(test_df['text'])
        
        print(f"Word vocabulary size: {len(word_vectorizer.vocabulary_)}")
        print(f"Char vocabulary size: {len(char_vectorizer.vocabulary_)}")
        
        print("Concatenating Word and Character Feature Matrices...")
        X_train = hstack([X_train_word, X_train_char]).tocsr()
        X_test = hstack([X_test_word, X_test_char]).tocsr()
        
        y_train = train_df['label'].values
        y_test = test_df['label'].values
        
        print(f"Combined Training Matrix Shape: {X_train.shape}")
        print(f"Combined Testing Matrix Shape: {X_test.shape}")
        
        # Setup directories for this fold
        fold_output_dir = os.path.join(output_dir, f"fold_{fold}")
        fold_models_dir = os.path.join(models_dir, f"fold_{fold}")
        os.makedirs(fold_output_dir, exist_ok=True)
        os.makedirs(fold_models_dir, exist_ok=True)
        
        # Save sparse feature matrices
        save_npz(os.path.join(fold_output_dir, "X_train.npz"), X_train)
        save_npz(os.path.join(fold_output_dir, "X_test.npz"), X_test)
        
        # Save target label arrays
        np.save(os.path.join(fold_output_dir, "y_train.npy"), y_train)
        np.save(os.path.join(fold_output_dir, "y_test.npy"), y_test)
        
        # Save indices of train/test for traceability
        np.save(os.path.join(fold_output_dir, "train_idx.npy"), train_idx)
        np.save(os.path.join(fold_output_dir, "test_idx.npy"), test_idx)
        
        # Save the fitted vectorizers using joblib
        joblib.dump(word_vectorizer, os.path.join(fold_models_dir, "word_vectorizer.joblib"))
        joblib.dump(char_vectorizer, os.path.join(fold_models_dir, "char_vectorizer.joblib"))
        
        print(f"Fold {fold} preprocessing completed successfully!")

if __name__ == "__main__":
    INPUT_CSV = "train_v4_drcat_01_cleaned.csv"
    OUTPUT_DIR = "processed_data"
    MODELS_DIR = "models"
    
    run_preprocessing(INPUT_CSV, OUTPUT_DIR, MODELS_DIR)
