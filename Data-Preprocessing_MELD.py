import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from collections import Counter

def combine_csv_files():
    """
    Combine the 3 MELD CSV files from the data folder into one CSV with only Utterance and Emotion columns
    """
    print("Combining MELD CSV files from data folder...")
    print("="*60)
    
    data_dir = 'data'
    csv_files = ['train_sent_emo_dya.csv', 'dev_sent_emo_dya.csv', 'test_sent_emo_dya.csv']
    
    combined_data = []
    
    for csv_file in csv_files:
        csv_path = os.path.join(data_dir, csv_file)
        if os.path.exists(csv_path):
            try:
                print(f"Loading {csv_file}...")
                df = pd.read_csv(csv_path)
                print(f"✅ Loaded {len(df)} records from {csv_file}")
                print(f"Columns: {list(df.columns)}")
                
                # Extract only Utterance and Emotion columns
                if 'Utterance' in df.columns and 'Emotion' in df.columns:
                    subset_df = df[['Utterance', 'Emotion']].copy()
                    combined_data.append(subset_df)
                    print(f"✅ Extracted {len(subset_df)} records with Utterance and Emotion columns")
                else:
                    print(f"❌ Required columns (Utterance, Emotion) not found in {csv_file}")
                    
            except Exception as e:
                print(f"❌ Error loading {csv_file}: {e}")
        else:
            print(f"⚠️ File not found: {csv_path}")
    
    if combined_data:
        # Combine all DataFrames
        final_df = pd.concat(combined_data, ignore_index=True)
        print(f"✅ Successfully combined {len(final_df):,} total records from all CSV files")
        return final_df
    else:
        print("❌ No data loaded from CSV files.")
        return None

def filter_joy_sadness(df):
    """
    Filter the dataset to only include Joy and Sadness emotions and convert to binary labels
    """
    print("\nFiltering for Joy and Sadness emotions only...")
    
    # Filter for only Joy and Sadness
    joy_sadness_df = df[df['Emotion'].isin(['joy', 'sadness'])].copy()
    
    print(f"✅ Filtered dataset contains {len(joy_sadness_df):,} records")
    print(f"Joy records: {len(joy_sadness_df[joy_sadness_df['Emotion'] == 'joy']):,}")
    print(f"Sadness records: {len(joy_sadness_df[joy_sadness_df['Emotion'] == 'sadness']):,}")
    
    # Convert to binary labels: Joy=1, Sadness=0
    print("\nConverting to binary labels: Joy=1, Sadness=0...")
    joy_sadness_df['label'] = (joy_sadness_df['Emotion'] == 'joy').astype(int)
    
    print(f"✅ Converted labels:")
    print(f"  - Joy (1): {len(joy_sadness_df[joy_sadness_df['label'] == 1]):,} records")
    print(f"  - Sadness (0): {len(joy_sadness_df[joy_sadness_df['label'] == 0]):,} records")
    
    return joy_sadness_df

def save_combined_csv(df, output_dir):
    """
    Save the combined DataFrame to CSV in the data folder
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'combined_meld_utterances_emotions.csv')
    
    print(f"\nSaving combined MELD dataset to {csv_path}...")
    df.to_csv(csv_path, index=False)
    print(f"✅ Successfully saved {len(df):,} records to {csv_path}")
    
    return csv_path

def save_joy_sadness_csv(df, output_dir):
    """
    Save the Joy/Sadness filtered DataFrame to CSV in the data folder
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'meld_joy_sadness_only.csv')
    
    print(f"\nSaving MELD Joy/Sadness dataset to {csv_path}...")
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
    fig.suptitle('MELD Emotion Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Emotion distribution bar chart
    emotion_counts = df['Emotion'].value_counts()
    
    # Create color map for emotions
    colors = plt.cm.Set3(np.linspace(0, 1, len(emotion_counts)))
    
    bars = axes[0, 0].bar(emotion_counts.index, emotion_counts.values, color=colors)
    axes[0, 0].set_title('Emotion Distribution', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Emotion', fontsize=12)
    axes[0, 0].set_ylabel('Count', fontsize=12)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for bar, count in zip(bars, emotion_counts.values):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, 
                       str(count), ha='center', va='bottom', fontweight='bold')
    
    # 2. Pie chart
    wedges, texts, autotexts = axes[0, 1].pie(emotion_counts.values, 
                                             labels=emotion_counts.index,
                                             autopct='%1.1f%%', 
                                             colors=colors, 
                                             startangle=90)
    axes[0, 1].set_title('Emotion Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    # 3. Text length distribution by emotion
    df['text_length'] = df['Utterance'].str.len()
    
    # Create box plot for text length by emotion
    emotion_text_lengths = []
    emotion_labels = []
    
    for emotion in emotion_counts.index:
        emotion_data = df[df['Emotion'] == emotion]['text_length']
        emotion_text_lengths.append(emotion_data)
        emotion_labels.append(emotion)
    
    box_plot = axes[1, 0].boxplot(emotion_text_lengths, tick_labels=emotion_labels, patch_artist=True)
    axes[1, 0].set_title('Text Length Distribution by Emotion', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Emotion', fontsize=12)
    axes[1, 0].set_ylabel('Text Length (characters)', fontsize=12)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Color the boxes
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 4. Word count distribution by emotion
    df['word_count'] = df['Utterance'].str.split().str.len()
    
    word_count_by_emotion = []
    for emotion in emotion_counts.index:
        emotion_data = df[df['Emotion'] == emotion]['word_count']
        word_count_by_emotion.append(emotion_data)
    
    box_plot2 = axes[1, 1].boxplot(word_count_by_emotion, tick_labels=emotion_labels, patch_artist=True)
    axes[1, 1].set_title('Word Count Distribution by Emotion', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Emotion', fontsize=12)
    axes[1, 1].set_ylabel('Word Count', fontsize=12)
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Color the boxes
    for patch, color in zip(box_plot2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.tight_layout()
    
    # Save the visualization
    viz_path = os.path.join(output_dir, 'meld_emotion_distribution_analysis.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"✅ Visualization saved to {viz_path}")
    plt.show()
    
    return emotion_counts

def create_detailed_statistics(df, emotion_counts, output_dir):
    """
    Create detailed statistics table
    """
    print("\n" + "="*80)
    print("MELD EMOTION DISTRIBUTION ANALYSIS")
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
        emotion_data = df[df['Emotion'] == emotion]
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
    stats_path = os.path.join(output_dir, 'meld_emotion_statistics.csv')
    stats_df.to_csv(stats_path, index=False)
    print(f"\n✅ Statistics table saved to {stats_path}")
    
    return stats_df

def main():
    """
    Main function to combine CSV files, filter for Joy/Sadness, and create visualizations
    """
    print("Starting MELD CSV Combination and Joy/Sadness Analysis...")
    print("="*60)
    
    # Set up paths
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    
    # Combine CSV files
    combined_df = combine_csv_files()
    if combined_df is None:
        print("❌ Failed to combine CSV files. Exiting...")
        return
    
    # Save combined CSV (all emotions)
    combined_csv_path = save_combined_csv(combined_df, output_dir)
    
    # Filter for Joy and Sadness only
    joy_sadness_df = filter_joy_sadness(combined_df)
    
    # Save Joy/Sadness CSV
    joy_sadness_csv_path = save_joy_sadness_csv(joy_sadness_df, output_dir)
    
    # Create visualizations for Joy/Sadness only
    emotion_counts = create_emotion_visualizations(joy_sadness_df, output_dir)
    
    # Create detailed statistics for Joy/Sadness
    stats_df = create_detailed_statistics(joy_sadness_df, emotion_counts, output_dir)
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Generated files:")
    print(f"  - {combined_csv_path} (all emotions)")
    print(f"  - {joy_sadness_csv_path} (Joy/Sadness only)")
    print(f"  - {os.path.join(output_dir, 'meld_emotion_distribution_analysis.png')}")
    print(f"  - {os.path.join(output_dir, 'meld_emotion_statistics.csv')}")
    
    print(f"\nDataset Summary:")
    print(f"  - Total records (all emotions): {len(combined_df):,}")
    print(f"  - Joy/Sadness records: {len(joy_sadness_df):,}")
    print(f"  - Joy records: {len(joy_sadness_df[joy_sadness_df['Emotion'] == 'joy']):,}")
    print(f"  - Sadness records: {len(joy_sadness_df[joy_sadness_df['Emotion'] == 'sadness']):,}")
    print(f"  - Unique emotions in filtered dataset: {len(emotion_counts)}")

if __name__ == "__main__":
    main()
