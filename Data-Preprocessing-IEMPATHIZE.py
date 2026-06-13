import pandas as pd
import os

def preprocess_iempathize():
    """
    Preprocess the IEmpathize dataset by removing label 0 (none) 
    and keeping only labels 1 (seeking-empathy) and 2 (providing-empathy).
    """
    
    # Load the IEmpathize dataset
    #df = pd.read_csv("data/IEmpathize_Dataset.csv")
    df = pd.read_csv("data/IEmpathize_Dataset.csv", encoding='latin-1')
    
    print(f"Original dataset size: {len(df)}")
    print(f"Label distribution:")
    print(df['label'].value_counts().sort_index())
    
    # Filter out label 0 (none) and keep only labels 1 and 2
    df_filtered = df[df['label'].isin([1, 2])]
    
    print(f"\nFiltered dataset size: {len(df_filtered)}")
    print(f"Filtered label distribution:")
    print(df_filtered['label'].value_counts().sort_index())
    
    # Remap labels: 1 (seeking-empathy) -> 0, 2 (providing-empathy) -> 1
    df_filtered['label'] = df_filtered['label'].map({1: 0, 2: 1})
    
    print(f"\nAfter label remapping:")
    print(f"Label distribution:")
    print(df_filtered['label'].value_counts().sort_index())
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save the filtered dataset
    output_file = "data/IEmpathize_Dataset_Filtered.csv"
    df_filtered.to_csv(output_file, index=False)
    
    print(f"\nSaved filtered dataset to: {output_file}")
    print(f"Final dataset contains {len(df_filtered)} samples:")
    print(f"- Seeking empathy (label 0): {len(df_filtered[df_filtered['label'] == 0])} samples")
    print(f"- Providing empathy (label 1): {len(df_filtered[df_filtered['label'] == 1])} samples")
    
    # Display sample data
    print("\nSample filtered data:")
    print(df_filtered.head(10))
    
    return df_filtered

if __name__ == "__main__":
    preprocess_iempathize()
