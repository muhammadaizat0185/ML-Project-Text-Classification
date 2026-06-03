import os
import pandas as pd
import numpy as np

def clean_dataset(input_path, output_path):
    print("=== Loading Raw Dataset ===")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}")
        
    df = pd.read_csv(input_path)
    initial_shape = df.shape
    print(f"Loaded {initial_shape[0]} rows and {initial_shape[1]} columns.")
    
    print("\n=== Preprocessing & Data Cleaning ===")
    
    # 1. Strip leading and trailing whitespace from text
    print("Normalizing whitespaces...")
    df['text'] = df['text'].astype(str).str.strip()
    
    # 2. Calculate word count for filtering
    print("Calculating word counts...")
    df['word_count'] = df['text'].apply(lambda x: len(x.split()))
    
    # 3. Identify and filter corrupted/ultra-short texts
    # We filter out text with fewer than 100 words as these are prompt titles, truncated lines, or gibberish.
    min_word_limit = 100
    corrupted_mask = df['word_count'] < min_word_limit
    num_corrupted = corrupted_mask.sum()
    
    print(f"Found {num_corrupted} rows with word count < {min_word_limit} (corrupted/truncated essays).")
    
    # Let's log some examples of filtered texts for transparency
    if num_corrupted > 0:
        print("\nExamples of filtered short/corrupted texts:")
        sample_corrupted = df[corrupted_mask].sort_values(by='word_count').head(5)
        for idx, row in sample_corrupted.iterrows():
            print(f" - [Label: {row['label']}, Source: {row['source']}, Words: {row['word_count']}] Text: {repr(row['text'][:150])}...")
            
    # Remove corrupted rows
    df_clean = df[~corrupted_mask].copy()
    
    # Drop the temporary word_count column if we want to match the original structure, 
    # but keeping it is actually very useful for exploratory analysis! 
    # Let's keep it for their model training and analysis.
    
    final_shape = df_clean.shape
    rows_removed = initial_shape[0] - final_shape[0]
    
    print(f"\nFiltered out {rows_removed} rows. Cleaned dataset has {final_shape[0]} rows.")
    
    # 4. Check label distribution in the cleaned dataset
    print("\n=== Cleaned Label Distribution ===")
    label_counts = df_clean['label'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / final_shape[0]) * 100
        label_name = "AI-Generated (1)" if label == 1 else "Human-Written (0)"
        print(f" - {label_name}: {count} ({percentage:.2f}%)")
        
    # 5. Save the cleaned dataset
    print(f"\n=== Saving Cleaned Dataset ===")
    df_clean.to_csv(output_path, index=False)
    print(f"Cleaned dataset successfully saved to: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    INPUT_FILE = "train_v4_drcat_01.csv"
    OUTPUT_FILE = "train_v4_drcat_01_cleaned.csv"
    
    clean_dataset(INPUT_FILE, OUTPUT_FILE)
