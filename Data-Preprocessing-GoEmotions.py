#!/usr/bin/env python3
"""
GoEmotions Joy/Sadness Analysis Script
Extracts datapoints that have either joy OR sadness (mutually exclusive)
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string

def clean_text(text):
    """
    Clean text by removing emojis, punctuation, and words less than 3 characters
    """
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string if not already
    text = str(text)
    
    # Remove emojis and emoticons
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002702-\U000027B0"  # dingbats
        u"\U000024C2-\U0001F251"  # enclosed characters
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub('', text)
    
    # Remove punctuation (keeping only letters, numbers, and spaces)
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove words less than 3 characters
    words = text.split()
    words = [word for word in words if len(word) >= 3]
    
    # Join words back together
    cleaned_text = ' '.join(words)
    
    # Strip leading/trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def load_and_combine_goemotions():
    """
    Load and combine all GoEmotions CSV files into one dataset
    """
    print("Loading GoEmotions dataset...")
    
    # Define the path to the data directory
    data_dir = "data_GoEmotions"
    
    # List all GoEmotions CSV files
    csv_files = [f for f in os.listdir(data_dir) if f.startswith('go_emotion') and f.endswith('.csv')]
    csv_files.sort()  # Sort to ensure consistent order
    
    print(f"Found {len(csv_files)} GoEmotions CSV files: {csv_files}")
    
    # Load and combine all CSV files
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(data_dir, file)
        print(f"Loading {file}...")
        df = pd.read_csv(file_path)
        dataframes.append(df)
    
    # Combine all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    
    print(f"Successfully combined {len(csv_files)} files")
    print(f"Total datapoints: {len(combined_df)}")
    
    return combined_df

def extract_joy_sadness_mutually_exclusive(df):
    """
    Extract datapoints that have either joy OR sadness (but NOT both)
    Create binary labels: 1 for joy, 0 for sadness
    """
    print("\n" + "="*60)
    print("EXTRACTING JOY/SADNESS MUTUALLY EXCLUSIVE DATAPOINTS")
    print("="*60)
    
    # Define masks for joy and sadness
    joy_mask = df['joy'] == 1
    sadness_mask = df['sadness'] == 1
    
    # Joy only: joy=1 AND sadness=0
    joy_only = joy_mask & ~sadness_mask
    
    # Sadness only: joy=0 AND sadness=1
    sadness_only = sadness_mask & ~joy_mask
    
    # Total mutually exclusive: either joy OR sadness (but not both)
    mutually_exclusive = joy_only | sadness_only
    
    # Count each category
    joy_only_count = joy_only.sum()
    sadness_only_count = sadness_only.sum()
    total_mutually_exclusive = mutually_exclusive.sum()
    
    print(f"Joy only (joy=1, sadness=0): {joy_only_count} datapoints")
    print(f"Sadness only (joy=0, sadness=1): {sadness_only_count} datapoints")
    print(f"Total mutually exclusive: {total_mutually_exclusive} datapoints")
    
    # Extract the mutually exclusive data
    mutually_exclusive_df = df[mutually_exclusive].copy()
    
    # Create binary labels: 1 for joy, 0 for sadness
    mutually_exclusive_df['label'] = 0  # Default to sadness (0)
    mutually_exclusive_df.loc[joy_only[mutually_exclusive], 'label'] = 1  # Joy gets label 1
    
    # Add emotion category labels for analysis
    mutually_exclusive_df['emotion_category'] = 'sadness'  # Default
    mutually_exclusive_df.loc[joy_only[mutually_exclusive], 'emotion_category'] = 'joy'
    
    # Verify the labeling
    print(f"\nVerification:")
    print(f"  Joy examples (label=1): {(mutually_exclusive_df['label'] == 1).sum()}")
    print(f"  Sadness examples (label=0): {(mutually_exclusive_df['label'] == 0).sum()}")
    print(f"  Total: {len(mutually_exclusive_df)}")
    
    return mutually_exclusive_df, joy_only_count, sadness_only_count, total_mutually_exclusive

def clean_dataset(df):
    """
    Clean the dataset by applying text cleaning to all text columns
    """
    print("\n" + "="*60)
    print("CLEANING DATASET")
    print("="*60)
    
    # Show before cleaning stats
    print(f"Before cleaning:")
    print(f"  Total datapoints: {len(df)}")
    print(f"  Average text length: {df['text'].str.len().mean():.1f} characters")
    
    # Apply text cleaning
    print("Applying text cleaning (removing emojis, punctuation, words < 3 chars)...")
    df['text_cleaned'] = df['text'].apply(clean_text)
    
    # Remove rows where cleaned text is empty or too short
    df_cleaned = df[df['text_cleaned'].str.len() >= 3].copy()
    
    # Replace original text with cleaned text
    df_cleaned['text'] = df_cleaned['text_cleaned']
    df_cleaned = df_cleaned.drop('text_cleaned', axis=1)
    
    # Show after cleaning stats
    print(f"After cleaning:")
    print(f"  Total datapoints: {len(df_cleaned)}")
    print(f"  Removed datapoints: {len(df) - len(df_cleaned)}")
    print(f"  Average text length: {df_cleaned['text'].str.len().mean():.1f} characters")
    
    # Show examples of cleaning
    print(f"\nText cleaning examples:")
    original_samples = df.head(3)
    cleaned_samples = df_cleaned.head(3)
    
    for i in range(min(3, len(original_samples))):
        if i < len(cleaned_samples):
            print(f"\nExample {i+1}:")
            print(f"  Original: {original_samples.iloc[i]['text']}")
            print(f"  Cleaned:  {cleaned_samples.iloc[i]['text']}")
    
    return df_cleaned

def create_visualizations(df, joy_count, sadness_count, total_count):
    """
    Create visualizations for the mutually exclusive joy/sadness dataset
    """
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # Set up the plotting style
    plt.style.use('default')
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GoEmotions: Joy vs Sadness (Mutually Exclusive)', fontsize=16, fontweight='bold')
    
    # 1. Bar chart of counts
    ax1 = axes[0, 0]
    categories = ['Joy Only', 'Sadness Only', 'Total']
    counts = [joy_count, sadness_count, total_count]
    colors = ['gold', 'lightblue', 'lightgreen']
    
    bars = ax1.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Count of Mutually Exclusive Datapoints')
    ax1.set_ylabel('Number of Datapoints')
    
    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie chart of distribution
    ax2 = axes[0, 1]
    sizes = [joy_count, sadness_count]
    labels = ['Joy Only', 'Sadness Only']
    colors_pie = ['gold', 'lightblue']
    
    wedges, texts, autotexts = ax2.pie(sizes, labels=labels, colors=colors_pie, autopct='%1.1f%%', 
                                      startangle=90, textprops={'fontsize': 10})
    ax2.set_title('Distribution of Joy vs Sadness')
    
    # 3. Text length distribution by emotion
    ax3 = axes[1, 0]
    
    # Calculate text lengths for each category
    joy_texts = df[df['emotion_category'] == 'joy']['text'].str.len()
    sadness_texts = df[df['emotion_category'] == 'sadness']['text'].str.len()
    
    ax3.hist(joy_texts, bins=30, alpha=0.7, label='Joy Only', color='gold', density=True)
    ax3.hist(sadness_texts, bins=30, alpha=0.7, label='Sadness Only', color='lightblue', density=True)
    ax3.set_title('Text Length Distribution by Emotion')
    ax3.set_xlabel('Character Count')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate statistics
    joy_texts_stats = df[df['emotion_category'] == 'joy']['text'].str.len()
    sadness_texts_stats = df[df['emotion_category'] == 'sadness']['text'].str.len()
    
    stats_text = f"""
    DATASET SUMMARY
    
    Total Mutually Exclusive Datapoints: {total_count:,}
    
    Joy Only: {joy_count:,} ({joy_count/total_count*100:.1f}%)
    - Avg text length: {joy_texts_stats.mean():.1f} chars
    - Median text length: {joy_texts_stats.median():.1f} chars
    
    Sadness Only: {sadness_count:,} ({sadness_count/total_count*100:.1f}%)
    - Avg text length: {sadness_texts_stats.mean():.1f} chars
    - Median text length: {sadness_texts_stats.median():.1f} chars
    
    Total Dataset Size: {len(df):,} datapoints
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('data_GoEmotions/joy_sadness_mutually_exclusive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Visualization saved to: data_GoEmotions/joy_sadness_mutually_exclusive_analysis.png")

def display_sample_texts(df):
    """
    Display sample texts from each emotion category
    """
    print("\n" + "="*60)
    print("SAMPLE TEXTS BY EMOTION CATEGORY")
    print("="*60)
    
    # Sample texts for joy
    joy_samples = df[df['emotion_category'] == 'joy']['text'].head(5)
    print(f"\nJOY ONLY SAMPLES ({len(joy_samples)} shown):")
    print("-" * 40)
    for i, text in enumerate(joy_samples, 1):
        print(f"{i}. {text}")
        print()
    
    # Sample texts for sadness
    sadness_samples = df[df['emotion_category'] == 'sadness']['text'].head(5)
    print(f"\nSADNESS ONLY SAMPLES ({len(sadness_samples)} shown):")
    print("-" * 40)
    for i, text in enumerate(sadness_samples, 1):
        print(f"{i}. {text}")
        print()

def save_filtered_dataset(df, filename="go_emotion_joy_sadness_mutually_exclusive.csv"):
    """
    Save the filtered dataset to a CSV file with only text and binary label
    """
    # Keep only essential columns: text and binary label
    simplified_df = df[['text', 'label']].copy()
    
    output_path = os.path.join("data_GoEmotions", filename)
    simplified_df.to_csv(output_path, index=False)
    print(f"\nSaved filtered dataset to: {output_path}")
    print(f"Columns: {list(simplified_df.columns)}")
    print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    
    # Show sample of the saved data
    print(f"\nSample of saved data:")
    print(simplified_df.head())

def main():
    """
    Main function to run the joy/sadness mutually exclusive analysis
    """
    print("GoEmotions Joy/Sadness Mutually Exclusive Analysis")
    print("="*60)
    
    # Load and combine all GoEmotions CSV files
    combined_df = load_and_combine_goemotions()
    
    # Extract mutually exclusive joy/sadness datapoints
    filtered_df, joy_count, sadness_count, total_count = extract_joy_sadness_mutually_exclusive(combined_df)
    
    # Clean the dataset
    cleaned_df = clean_dataset(filtered_df)
    
    # Update counts after cleaning
    joy_count_cleaned = (cleaned_df['label'] == 1).sum()
    sadness_count_cleaned = (cleaned_df['label'] == 0).sum()
    total_count_cleaned = len(cleaned_df)
    
    # Create visualizations
    create_visualizations(cleaned_df, joy_count_cleaned, sadness_count_cleaned, total_count_cleaned)
    
    # Display sample texts
    display_sample_texts(cleaned_df)
    
    # Save the cleaned dataset
    save_filtered_dataset(cleaned_df)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("Summary:")
    print(f"  - Original mutually exclusive datapoints: {total_count:,}")
    print(f"  - After cleaning: {total_count_cleaned:,}")
    print(f"  - Joy examples (label=1): {joy_count_cleaned:,}")
    print(f"  - Sadness examples (label=0): {sadness_count_cleaned:,}")
    print(f"  - Files created:")
    print(f"    * data_GoEmotions/go_emotion_joy_sadness_mutually_exclusive.csv")
    print(f"    * data_GoEmotions/joy_sadness_mutually_exclusive_analysis.png")

if __name__ == "__main__":
    main()