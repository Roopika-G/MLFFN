import pandas as pd
import os
import re

def clean_text(text):
    """
    Clean text by removing numbers, short words, brackets, and other unwanted characters.
    """
    if pd.isna(text):
        return ""
    
    # Convert to string
    text = str(text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove numbers (including decimals)
    text = re.sub(r'\d+\.?\d*', '', text)
    
    # Remove brackets and their contents: [], (), {}
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\'\"\-]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Split into words and filter
    words = text.split()
    
    # Keep only words with 3+ characters (you can change this to 2 if you prefer)
    cleaned_words = [word for word in words if len(word) >= 3]
    
    # Join back into text
    cleaned_text = ' '.join(cleaned_words)
    
    return cleaned_text.strip()

def preprocess_reddit_data():
    """
    Combines emotional-reactions, explorations, and interpretations CSV files,
    filters out scores 0 and 2, keeping only score 1.
    Creates a new CSV with 'text' and 'score' columns.
    """
    
    # File paths
    files = [
        "data/emotional-reactions-reddit.csv",
        "data/explorations-reddit.csv", 
        "data/interpretations-reddit.csv"
    ]
    
    # Load and combine all CSV files
    combined_data = []
    
    for file_path in files:
        if os.path.exists(file_path):
            print(f"Loading {file_path}...")
            df = pd.read_csv(file_path)
            
            # Extract text (response_post) and score (level)
            df_subset = df[['response_post', 'level']].copy()
            df_subset.columns = ['text', 'score']
            
            combined_data.append(df_subset)
            print(f"  - Loaded {len(df_subset)} samples")
        else:
            print(f"Warning: {file_path} not found!")
    
    if not combined_data:
        print("No data files found!")
        return
    
    # Combine all data
    df_combined = pd.concat(combined_data, ignore_index=True)
    print(f"\nTotal combined samples: {len(df_combined)}")
    
    # Show score distribution before filtering
    print(f"\nScore distribution before filtering:")
    print(df_combined['score'].value_counts().sort_index())
    
    # Filter out score 1, keep only scores 0 and 2
    df_filtered = df_combined[df_combined['score'].isin([0, 2])].copy()
    
    print(f"\nAfter filtering (keeping only scores 0 and 2):")
    print(f"Filtered samples: {len(df_filtered)}")
    print(f"Score distribution:")
    print(df_filtered['score'].value_counts().sort_index())
    
    # Remap scores: 0 → 0, 2 → 1
    df_filtered['score'] = df_filtered['score'].map({0: 0, 2: 1})
    
    print(f"\nAfter remapping (0→0, 2→1):")
    print(f"Final samples: {len(df_filtered)}")
    print(f"Score distribution:")
    print(df_filtered['score'].value_counts().sort_index())
    
    # Clean the text data
    print(f"\nCleaning text data...")
    df_filtered['text'] = df_filtered['text'].apply(clean_text)
    
    # Remove rows with empty text after cleaning
    original_count = len(df_filtered)
    df_filtered = df_filtered[df_filtered['text'].str.strip() != ''].copy()
    removed_count = original_count - len(df_filtered)
    
    print(f"Removed {removed_count} samples with empty text after cleaning")
    print(f"Final samples after text cleaning: {len(df_filtered)}")
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save the filtered dataset
    output_file = "data/EPITOME_filtered_binary.csv"
    df_filtered.to_csv(output_file, index=False)
    
    print(f"\n✅ Saved filtered dataset to: {output_file}")
    print(f"Final dataset contains {len(df_filtered)} samples with binary labels (0 and 1)")
    
    # Display sample data (before and after cleaning)
    print("\nSample data before cleaning:")
    sample_before = df_combined[df_combined['score'].isin([0, 2])].head(3)
    for i, text in enumerate(sample_before['response_post']):
        print(f"  {i+1}. {text[:100]}...")
    
    print("\nSample data after cleaning:")
    print(df_filtered.head(3))
    
    # Show text length statistics
    text_lengths = df_filtered['text'].str.len()
    print(f"\nText length statistics:")
    print(f"  - Mean length: {text_lengths.mean():.1f} characters")
    print(f"  - Median length: {text_lengths.median():.1f} characters")
    print(f"  - Min length: {text_lengths.min()} characters")
    print(f"  - Max length: {text_lengths.max()} characters")
    
    return df_filtered

if __name__ == "__main__":
    preprocess_reddit_data()
