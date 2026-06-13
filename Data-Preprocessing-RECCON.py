import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import Counter
import numpy as np

def load_json_files(data_dir):
    """
    Load all JSON files from the data directory
    """
    json_files = []
    json_file_paths = [
        'dailydialog_train.json',
        'dailydialog_test.json', 
        'dailydialog_valid.json',
        'iemocap_test.json'
    ]
    
    for file_name in json_file_paths:
        file_path = os.path.join(data_dir, file_name)
        if os.path.exists(file_path):
            print(f"Loading {file_name}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_files.append((file_name, data))
        else:
            print(f"Warning: {file_name} not found in {data_dir}")
    
    return json_files

def flatten_conversations(json_files):
    """
    Flatten all conversations from JSON files into a list of records
    """
    all_records = []
    
    for file_name, data in json_files:
        print(f"Processing {file_name}...")
        
        for conversation_id, conversations in data.items():
            # Each conversation_id can have multiple conversation instances
            for conv_idx, conversation in enumerate(conversations):
                for turn_data in conversation:
                    record = {
                        'file_source': file_name,
                        'conversation_id': conversation_id,
                        'conversation_index': conv_idx,
                        'turn': turn_data['turn'],
                        'speaker': turn_data['speaker'],
                        'utterance': turn_data['utterance'],
                        'emotion': turn_data['emotion']
                    }
                    
                    # Add optional fields if they exist
                    if 'expanded emotion cause evidence' in turn_data:
                        record['expanded_emotion_cause_evidence'] = str(turn_data['expanded emotion cause evidence'])
                    else:
                        record['expanded_emotion_cause_evidence'] = None
                        
                    if 'expanded emotion cause span' in turn_data:
                        record['expanded_emotion_cause_span'] = str(turn_data['expanded emotion cause span'])
                    else:
                        record['expanded_emotion_cause_span'] = None
                        
                    if 'type' in turn_data:
                        record['type'] = str(turn_data['type'])
                    else:
                        record['type'] = None
                    
                    all_records.append(record)
    
    return all_records

def create_visualizations(df, output_dir):
    """
    Create visualizations for emotion distribution by turn
    """
    plt.style.use('default')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Emotion distribution for Turn 1
    plt.figure(figsize=(12, 8))
    turn1_emotions = df[df['turn'] == 1]['emotion'].value_counts()
    
    plt.subplot(2, 2, 1)
    turn1_emotions.plot(kind='bar', color='skyblue')
    plt.title('Emotion Distribution - Turn 1', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, v in enumerate(turn1_emotions.values):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 2. Emotion distribution for Turn 2
    plt.subplot(2, 2, 2)
    turn2_emotions = df[df['turn'] == 2]['emotion'].value_counts()
    
    turn2_emotions.plot(kind='bar', color='lightcoral')
    plt.title('Emotion Distribution - Turn 2', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, v in enumerate(turn2_emotions.values):
        plt.text(i, v + 0.5, str(v), ha='center', va='bottom', fontweight='bold')
    
    # 3. Side-by-side comparison
    plt.subplot(2, 2, 3)
    
    # Get all unique emotions
    all_emotions = sorted(df['emotion'].unique())
    
    turn1_counts = [turn1_emotions.get(emotion, 0) for emotion in all_emotions]
    turn2_counts = [turn2_emotions.get(emotion, 0) for emotion in all_emotions]
    
    x = np.arange(len(all_emotions))
    width = 0.35
    
    plt.bar(x - width/2, turn1_counts, width, label='Turn 1', color='skyblue', alpha=0.8)
    plt.bar(x + width/2, turn2_counts, width, label='Turn 2', color='lightcoral', alpha=0.8)
    
    plt.title('Emotion Distribution Comparison: Turn 1 vs Turn 2', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(x, all_emotions, rotation=45)
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # 4. Pie chart for overall emotion distribution
    plt.subplot(2, 2, 4)
    overall_emotions = df['emotion'].value_counts()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(overall_emotions)))
    wedges, texts, autotexts = plt.pie(overall_emotions.values, labels=overall_emotions.index, 
                                      autopct='%1.1f%%', colors=colors, startangle=90)
    
    plt.title('Overall Emotion Distribution', fontsize=14, fontweight='bold')
    
    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'RECCON_Dataset_Emotion_Analysis.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a detailed statistics table
    print("\n" + "="*80)
    print("RECCON DATASET EMOTION ANALYSIS")
    print("="*80)
    
    print(f"\nTotal Records: {len(df):,}")
    print(f"Total Conversations: {df['conversation_id'].nunique():,}")
    print(f"Unique Emotions: {df['emotion'].nunique()}")
    
    print(f"\nTurn 1 Statistics:")
    print(f"  Total Turn 1 records: {len(df[df['turn'] == 1]):,}")
    print(f"  Unique emotions in Turn 1: {df[df['turn'] == 1]['emotion'].nunique()}")
    
    print(f"\nTurn 2 Statistics:")
    print(f"  Total Turn 2 records: {len(df[df['turn'] == 2]):,}")
    print(f"  Unique emotions in Turn 2: {df[df['turn'] == 2]['emotion'].nunique()}")
    
    print(f"\nEmotion Distribution by Turn:")
    print("-" * 50)
    
    # Create a comparison table
    comparison_df = pd.DataFrame({
        'Emotion': all_emotions,
        'Turn_1_Count': turn1_counts,
        'Turn_2_Count': turn2_counts,
        'Total_Count': [turn1_counts[i] + turn2_counts[i] for i in range(len(all_emotions))]
    })
    
    comparison_df['Turn_1_Percentage'] = (comparison_df['Turn_1_Count'] / comparison_df['Turn_1_Count'].sum() * 100).round(2)
    comparison_df['Turn_2_Percentage'] = (comparison_df['Turn_2_Count'] / comparison_df['Turn_2_Count'].sum() * 100).round(2)
    
    print(comparison_df.to_string(index=False))
    
    # Save the comparison table
    comparison_df.to_csv(os.path.join(output_dir, 'RECCON_Emotion_Comparison_Table.csv'), index=False)
    
    return comparison_df

def main():
    """
    Main function to process JSON files and create CSV output
    """
    # Set up paths
    data_dir = 'data'
    output_dir = 'data'
    
    print("Starting RECCON Dataset Processing...")
    print("="*50)
    
    # Load JSON files
    json_files = load_json_files(data_dir)
    
    if not json_files:
        print("No JSON files found! Please check the data directory.")
        return
    
    # Flatten conversations
    print("\nFlattening conversations...")
    all_records = flatten_conversations(json_files)
    
    # Create DataFrame
    print(f"\nCreating DataFrame with {len(all_records):,} records...")
    df = pd.DataFrame(all_records)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'RECCON_Dataset.csv')
    print(f"Saving to {csv_path}...")
    df.to_csv(csv_path, index=False)
    
    print(f"✅ Successfully saved {len(df):,} records to {csv_path}")
    
    # Create visualizations
    print("\nCreating visualizations...")
    comparison_df = create_visualizations(df, output_dir)
    
    print(f"\n✅ Processing complete!")
    print(f"📊 Generated visualizations saved to {output_dir}/")
    print(f"📈 Emotion comparison table saved to {output_dir}/RECCON_Emotion_Comparison_Table.csv")
    
    # Display basic info about the dataset
    print(f"\nDataset Summary:")
    print(f"  - Total records: {len(df):,}")
    print(f"  - Total conversations: {df['conversation_id'].nunique():,}")
    print(f"  - Unique emotions: {df['emotion'].nunique()}")
    print(f"  - Turn range: {df['turn'].min()} to {df['turn'].max()}")
    print(f"  - Files processed: {df['file_source'].nunique()}")

if __name__ == "__main__":
    main()
