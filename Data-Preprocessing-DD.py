import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from collections import Counter

def load_dailydialog_from_parquet():
    """
    Load the DailyDialog dataset from parquet files in the data folder
    """
    print("Loading DailyDialog dataset from parquet files...")
    print("="*60)
    
    data_dir = 'data'
    parquet_files = ['0000.parquet', '0000 (1).parquet', '0000 (2).parquet']
    
    combined_data = []
    
    for parquet_file in parquet_files:
        parquet_path = os.path.join(data_dir, parquet_file)
        if os.path.exists(parquet_path):
            try:
                print(f"Loading {parquet_file}...")
                df = pd.read_parquet(parquet_path)
                print(f"✅ Loaded {len(df)} records from {parquet_file}")
                print(f"Columns: {list(df.columns)}")
                
                # Convert to list of dictionaries
                for _, row in df.iterrows():
                    combined_data.append({
                        'dialog': row['utterances'],
                        'act': row['acts'],
                        'emotion': row['emotions'],
                        'topic': row['id']  # Using id as topic placeholder
                    })
                    
            except Exception as e:
                print(f"❌ Error loading {parquet_file}: {e}")
        else:
            print(f"⚠️ File not found: {parquet_path}")
    
    if combined_data:
        print(f"✅ Successfully loaded {len(combined_data)} total records from parquet files")
        return combined_data
    else:
        print("❌ No data loaded from parquet files. Using sample data...")
        
        # Create sample data structure similar to the DailyDialog dataset
        sample_data = [
            {
                "dialog": ["Hello, how are you today?", "I'm feeling great, thanks for asking!", "That's wonderful to hear!"],
                "act": [1, 1, 1],
                "emotion": [0, 4, 4],  # 0: no emotion, 4: happiness
                "topic": 1
            },
            {
                "dialog": ["I'm so sad about what happened.", "I understand how you feel.", "It's really difficult."],
                "act": [1, 1, 1],
                "emotion": [5, 0, 5],  # 5: sadness
                "topic": 4
            }
        ]
        print("Using sample data for demonstration...")
        return sample_data

def convert_to_dataframe(dataset):
    """
    Convert the dataset to pandas DataFrame
    """
    print("\nConverting dataset to pandas DataFrame...")
    
    # Handle list of dictionaries (most common format)
    if isinstance(dataset, list) and len(dataset) > 0 and isinstance(dataset[0], dict):
        df = pd.DataFrame(dataset)
    elif isinstance(dataset, dict) and 'dialog' in dataset and 'emotion' in dataset:
        # Direct data format
        df = pd.DataFrame(dataset)
    else:
        # If it's a list of records, convert to DataFrame
        df = pd.DataFrame(dataset)
    
    print(f"✅ Converted to DataFrame with {len(df):,} records")
    print(f"Columns: {list(df.columns)}")
    print(f"Sample data:")
    print(df.head())
    
    return df

def extract_happiness_sadness_utterances(df):
    """
    Extract happiness (emotion=4) and sadness (emotion=5) utterances from DailyDialog
    """
    print("\nExtracting happiness and sadness utterances...")
    
    # Flatten the dialog data to get individual utterances
    all_utterances = []
    
    for idx, row in df.iterrows():
        dialog = row['dialog']
        emotions = row['emotion']
        acts = row['act']
        topic = row['topic']
        
        # Ensure all lists have the same length
        if isinstance(dialog, list) and isinstance(emotions, list):
            min_length = min(len(dialog), len(emotions))
            for i in range(min_length):
                utterance = dialog[i]
                emotion = emotions[i]
                act = acts[i] if isinstance(acts, list) and i < len(acts) else acts
                
                # Only keep happiness (4) and sadness (5) utterances
                if emotion in [4, 5]:
                    all_utterances.append({
                        'conversation': utterance,
                        'emotion': emotion,
                        'act': act,
                        'topic': topic,
                        'dialog_id': idx
                    })
    
    # Convert to DataFrame
    utterances_df = pd.DataFrame(all_utterances)
    
    # Map emotion numbers to names
    emotion_map = {4: 'happiness', 5: 'sadness'}
    utterances_df['emotion_name'] = utterances_df['emotion'].map(emotion_map)
    
    # Convert to binary labels: happiness=1, sadness=0
    utterances_df['label'] = (utterances_df['emotion'] == 4).astype(int)
    
    print(f"✅ Extracted {len(utterances_df):,} happiness and sadness utterances")
    print(f"Happiness utterances: {len(utterances_df[utterances_df['emotion'] == 4]):,}")
    print(f"Sadness utterances: {len(utterances_df[utterances_df['emotion'] == 5]):,}")
    
    return utterances_df

def save_to_csv(df, output_dir):
    """
    Save the DataFrame to CSV in the data folder
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'dailydialog_happiness_sadness.csv')
    
    print(f"\nSaving dataset to {csv_path}...")
    df.to_csv(csv_path, index=False)
    print(f"✅ Successfully saved {len(df):,} records to {csv_path}")
    
    return csv_path

def create_emotion_visualizations(df, output_dir):
    """
    Create comprehensive visualizations for emotion distribution
    """
    print("\nCreating emotion distribution visualizations...")
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('DailyDialog Happiness vs Sadness Analysis', fontsize=16, fontweight='bold')
    
    # 1. Emotion distribution bar chart
    emotion_counts = df['emotion_name'].value_counts()
    
    axes[0, 0].bar(emotion_counts.index, emotion_counts.values, 
                   color=['#4ECDC4', '#FF6B6B'])  # Teal for happiness, Red for sadness
    axes[0, 0].set_title('Happiness vs Sadness Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Emotion', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, v in enumerate(emotion_counts.values):
        axes[0, 0].text(i, v + 50, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie chart
    colors = ['#4ECDC4', '#FF6B6B']  # Teal for happiness, Red for sadness
    wedges, texts, autotexts = axes[0, 1].pie(emotion_counts.values, 
                                             labels=emotion_counts.index,
                                             autopct='%1.1f%%', 
                                             colors=colors, 
                                             startangle=90)
    axes[0, 1].set_title('Happiness vs Sadness (Percentage)', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(12)
    
    # 3. Text length distribution by emotion
    df['text_length'] = df['conversation'].str.len()
    
    # Create box plot for text length by emotion
    emotion_text_lengths = []
    emotion_labels = []
    
    for emotion in emotion_counts.index:
        emotion_data = df[df['emotion_name'] == emotion]['text_length']
        emotion_text_lengths.append(emotion_data)
        emotion_labels.append(emotion)
    
    box_plot = axes[1, 0].boxplot(emotion_text_lengths, tick_labels=emotion_labels, patch_artist=True)
    axes[1, 0].set_title('Text Length Distribution by Emotion', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Emotion', fontsize=12)
    axes[1, 0].set_ylabel('Text Length (characters)', fontsize=12)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 4. Word count distribution by emotion
    df['word_count'] = df['conversation'].str.split().str.len()
    
    word_count_by_emotion = []
    for emotion in emotion_counts.index:
        emotion_data = df[df['emotion_name'] == emotion]['word_count']
        word_count_by_emotion.append(emotion_data)
    
    box_plot2 = axes[1, 1].boxplot(word_count_by_emotion, tick_labels=emotion_labels, patch_artist=True)
    axes[1, 1].set_title('Word Count Distribution by Emotion', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Emotion', fontsize=12)
    axes[1, 1].set_ylabel('Word Count', fontsize=12)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Color the boxes
    for patch, color in zip(box_plot2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.tight_layout()
    
    # Save the visualization
    viz_path = os.path.join(output_dir, 'dailydialog_happiness_sadness_analysis.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {viz_path}")
    plt.show()
    
    return emotion_counts

def create_detailed_statistics(df, emotion_counts, output_dir):
    """
    Create detailed statistics table
    """
    print("\n" + "="*80)
    print("DAILYDIALOG HAPPINESS vs SADNESS ANALYSIS")
    print("="*80)
    
    print(f"\nTotal Records: {len(df):,}")
    print(f"Unique Emotions: {len(emotion_counts)}")
    print(f"Average Text Length: {df['text_length'].mean():.1f} characters")
    print(f"Average Word Count: {df['word_count'].mean():.1f} words")
    
    print(f"\nEmotion Distribution:")
    print("-" * 50)
    
    # Create detailed statistics table
    stats_data = []
    for emotion, count in emotion_counts.items():
        emotion_data = df[df['emotion_name'] == emotion]
        percentage = (count / len(df)) * 100
        avg_length = emotion_data['text_length'].mean()
        avg_words = emotion_data['word_count'].mean()
        
        stats_data.append({
            'Emotion': emotion,
            'Count': count,
            'Percentage': f"{percentage:.2f}%",
            'Avg_Text_Length': f"{avg_length:.1f}",
            'Avg_Word_Count': f"{avg_words:.1f}"
        })
    
    stats_df = pd.DataFrame(stats_data)
    print(stats_df.to_string(index=False))
    
    # Save statistics table
    stats_path = os.path.join(output_dir, 'dailydialog_happiness_sadness_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✅ Statistics table saved to {stats_path}")
    
    return stats_df

def main():
    """
    Main function to process the DailyDialog dataset from parquet files
    """
    print("Starting DailyDialog Dataset Processing from Parquet Files...")
    print("="*60)
    
    # Set up paths
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset from parquet files
    dataset = load_dailydialog_from_parquet()
    if dataset is None:
        print("❌ Failed to load dataset. Exiting...")
        return
    
    # Convert to DataFrame
    df = convert_to_dataframe(dataset)
    
    # Extract happiness and sadness utterances
    happiness_sadness_df = extract_happiness_sadness_utterances(df)
    
    # Save to CSV
    csv_path = save_to_csv(happiness_sadness_df, output_dir)
    
    # Create visualizations
    emotion_counts = create_emotion_visualizations(happiness_sadness_df, output_dir)
    
    # Create detailed statistics
    stats_df = create_detailed_statistics(happiness_sadness_df, emotion_counts, output_dir)
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Generated files:")
    print(f"  - {csv_path}")
    print(f"  - {os.path.join(output_dir, 'dailydialog_happiness_sadness_analysis.png')}")
    print(f"  - {os.path.join(output_dir, 'dailydialog_happiness_sadness_statistics.csv')}")
    
    print(f"\nDataset Summary:")
    print(f"  - Total records: {len(happiness_sadness_df):,}")
    print(f"  - Happiness records: {len(happiness_sadness_df[happiness_sadness_df['emotion'] == 4]):,}")
    print(f"  - Sadness records: {len(happiness_sadness_df[happiness_sadness_df['emotion'] == 5]):,}")
    print(f"  - Unique emotions: {len(emotion_counts)}")

if __name__ == "__main__":
    main()
