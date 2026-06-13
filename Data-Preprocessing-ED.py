from datasets import load_dataset
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

def create_visualizations(df):
    """Create comprehensive visualizations for ED dataset"""
    print("\n============================================================")
    print("CREATING VISUALIZATIONS")
    print("============================================================")
    
    # Create data_ED directory
    os.makedirs("data_ED", exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Emotion Distribution
    plt.figure(figsize=(12, 8))
    
    # Subplot 1: Bar chart
    plt.subplot(2, 2, 1)
    emotion_counts = df['emotion'].value_counts()
    colors = ['#FF6B6B', '#4ECDC4']
    bars = plt.bar(emotion_counts.index, emotion_counts.values, color=colors)
    plt.title('Emotion Distribution (ED Dataset)', fontsize=14, fontweight='bold')
    plt.xlabel('Emotion', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, count in zip(bars, emotion_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 2: Pie chart
    plt.subplot(2, 2, 2)
    plt.pie(emotion_counts.values, labels=emotion_counts.index, autopct='%1.1f%%',
            colors=colors, startangle=90)
    plt.title('Emotion Distribution (Percentage)', fontsize=14, fontweight='bold')
    
    # Subplot 3: Binary label distribution
    plt.subplot(2, 2, 3)
    label_counts = df['label'].value_counts()
    label_names = ['Sad (0)', 'Joyful (1)']
    colors_binary = ['#FF6B6B', '#4ECDC4']
    bars = plt.bar(label_names, label_counts.values, color=colors_binary)
    plt.title('Binary Label Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Label', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    # Add value labels
    for bar, count in zip(bars, label_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f'{count:,}', ha='center', va='bottom', fontweight='bold')
    
    # Subplot 4: Text length distribution
    plt.subplot(2, 2, 4)
    df['text_length'] = df['conversation'].str.len()
    plt.hist(df['text_length'], bins=50, color='skyblue', alpha=0.7, edgecolor='black')
    plt.title('Conversation Length Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Character Count', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(df['text_length'].mean(), color='red', linestyle='--', 
                label=f'Mean: {df["text_length"].mean():.0f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('data_ED/ed_emotion_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved to: data_ED/ed_emotion_analysis.png")
    
    # 2. Sample texts visualization
    create_sample_texts_visualization(df)
    
    # 3. Statistics
    create_statistics_file(df)

def create_sample_texts_visualization(df):
    """Create visualization showing sample texts by emotion"""
    plt.figure(figsize=(15, 10))
    
    # Get samples
    joyful_samples = df[df['emotion'] == 'joyful']['conversation'].head(5).tolist()
    sad_samples = df[df['emotion'] == 'sad']['conversation'].head(5).tolist()
    
    # Create text display
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Joyful samples
    axes[0].text(0.05, 0.95, 'JOYFUL SAMPLES:', transform=axes[0].transAxes, 
                fontsize=16, fontweight='bold', color='green')
    for i, text in enumerate(joyful_samples):
        axes[0].text(0.05, 0.85 - i*0.15, f"{i+1}. {text[:100]}...", 
                    transform=axes[0].transAxes, fontsize=10, wrap=True)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)
    axes[0].axis('off')
    
    # Sad samples
    axes[1].text(0.05, 0.95, 'SAD SAMPLES:', transform=axes[1].transAxes, 
                fontsize=16, fontweight='bold', color='red')
    for i, text in enumerate(sad_samples):
        axes[1].text(0.05, 0.85 - i*0.15, f"{i+1}. {text[:100]}...", 
                    transform=axes[1].transAxes, fontsize=10, wrap=True)
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('data_ED/ed_sample_texts.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Sample texts visualization saved to: data_ED/ed_sample_texts.png")

def create_statistics_file(df):
    """Create detailed statistics file"""
    stats = {
        'Dataset': 'Empathetic Dialogues (ED)',
        'Total_Records': len(df),
        'Joyful_Records': len(df[df['emotion'] == 'joyful']),
        'Sad_Records': len(df[df['emotion'] == 'sad']),
        'Joyful_Percentage': (len(df[df['emotion'] == 'joyful']) / len(df)) * 100,
        'Sad_Percentage': (len(df[df['emotion'] == 'sad']) / len(df)) * 100,
        'Average_Text_Length': df['conversation'].str.len().mean(),
        'Median_Text_Length': df['conversation'].str.len().median(),
        'Min_Text_Length': df['conversation'].str.len().min(),
        'Max_Text_Length': df['conversation'].str.len().max(),
        'Unique_Conversations': df['conv_id'].nunique()
    }
    
    # Save statistics
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv('data_ED/ed_statistics.csv', index=False)
    
    print("Statistics saved to: data_ED/ed_statistics.csv")
    
    # Print summary
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Total records: {stats['Total_Records']:,}")
    print(f"  - Joyful records: {stats['Joyful_Records']:,} ({stats['Joyful_Percentage']:.1f}%)")
    print(f"  - Sad records: {stats['Sad_Records']:,} ({stats['Sad_Percentage']:.1f}%)")
    print(f"  - Average text length: {stats['Average_Text_Length']:.1f} characters")
    print(f"  - Unique conversations: {stats['Unique_Conversations']:,}")

def preprocess():
    print("Empathetic Dialogues (ED) Joy/Sadness Analysis")
    print("============================================================")
    
    # Load empathetic dialogues dataset from HuggingFace
    print("Loading Empathetic Dialogues dataset...")
    dataset = load_dataset("Estwld/empathetic_dialogues_llm")

    # Convert all splits to pandas DataFrame and combine
    print("Loading all splits (train, validation, test)...")
    df_train = dataset["train"].to_pandas()
    df_validation = dataset["valid"].to_pandas() 
    df_test = dataset["test"].to_pandas()

    # Combine all splits
    df_combined = pd.concat([df_train, df_validation, df_test], ignore_index=True)
    
    print(f"Total samples in dataset: {len(df_combined)}")
    print(f"Available emotions: {df_combined['emotion'].unique()}")
    print(f"Dataset columns: {list(df_combined.columns)}")

    # Filter dataset for only Joyful and Sad emotions (correct emotion names)
    df_filtered = df_combined[df_combined["emotion"].isin(["joyful", "sad"])]
    print(f"Filtered samples (joyful + sad): {len(df_filtered)}")

    # Extract conversations from the conversations column and combine into full dialogue
    processed_conversations = []
    
    for _, row in df_filtered.iterrows():
        # Extract all dialogue turns from the conversations list
        full_dialogue = []
        for turn in row['conversations']:
            full_dialogue.append(turn['content'])
        
        # Join all dialogue turns into one text
        combined_text = " ".join(full_dialogue)
        
        processed_conversations.append({
            'conv_id': row['conv_id'],
            'emotion': row['emotion'],
            'situation': row['situation'],
            'conversation': combined_text  # Full dialogue as one text
        })
    
    # Convert to DataFrame
    conversations_df = pd.DataFrame(processed_conversations)
    
    print(f"Number of conversations: {len(conversations_df)}")
    print(f"Joyful conversations: {len(conversations_df[conversations_df['emotion'] == 'joyful'])}")
    print(f"Sad conversations: {len(conversations_df[conversations_df['emotion'] == 'sad'])}")

    # Convert to binary labels: Joyful=1, Sad=0
    print("\nConverting to binary labels: Joyful=1, Sad=0...")
    conversations_df['label'] = (conversations_df['emotion'] == 'joyful').astype(int)

    print(f"✅ Converted labels:")
    print(f"  - Joyful (1): {len(conversations_df[conversations_df['label'] == 1]):,} records")
    print(f"  - Sad (0): {len(conversations_df[conversations_df['label'] == 0]):,} records")

    # Create visualizations
    create_visualizations(conversations_df)

    # Create data_ED directory if it doesn't exist
    os.makedirs("data_ED", exist_ok=True)
    
    # Save FULL CSV with all columns (for reference and visualizations)
    conversations_df.to_csv("data_ED/empathetic_dialogues_full.csv", index=False)
    print("Saved data_ED/empathetic_dialogues_full.csv with", len(conversations_df), "rows (all columns).")
    
    # Create simplified CSV with only Utterance and Emotion (binary 0/1) columns for training
    final_df = pd.DataFrame({
        'Utterance': conversations_df['conversation'],
        'Emotion': conversations_df['label']  # Binary labels: 0=Sad, 1=Joyful
    })
    
    # Save simplified CSV file in data_ED/ directory
    final_df.to_csv("data_ED/empathetic_dialogues.csv", index=False)
    print("Saved data_ED/empathetic_dialogues.csv with", len(final_df), "rows (Utterance + Emotion only).")
    print(f"Columns: {list(final_df.columns)} - Emotion values: 0 (Sad), 1 (Joyful)")
    
    # Display sample data
    print("\nSample conversations:")
    print(conversations_df.head())
    
    # Show a sample conversation text
    print("\nSample conversation text:")
    if len(conversations_df) > 0:
        print(f"Emotion: {conversations_df.iloc[0]['emotion']}")
        print(f"Conversation: {conversations_df.iloc[0]['conversation'][:200]}...")
    
    print("\n============================================================")
    print("ANALYSIS COMPLETE!")
    print("============================================================")
    print("Summary:")
    print(f"  - Total conversations: {len(conversations_df):,}")
    print(f"  - Joyful conversations: {len(conversations_df[conversations_df['emotion'] == 'joyful']):,}")
    print(f"  - Sad conversations: {len(conversations_df[conversations_df['emotion'] == 'sad']):,}")
    print("  - Files created:")
    print("    * data_ED/empathetic_dialogues.csv")
    print("    * data_ED/ed_emotion_analysis.png")
    print("    * data_ED/ed_sample_texts.png")
    print("    * data_ED/ed_statistics.csv")

if __name__ == "__main__":
    preprocess()
