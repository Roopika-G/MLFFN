# util.py
# Utility functions for loading data and splitting into train/dev/test sets.

import pandas as pd
import constants as cs
import numpy as np

def get_conversations():
    """
    Loads the ED dataset with Joy/Sadness binary labels.
    Returns a pandas DataFrame with columns: [Utterance, Emotion]
    """
    df = pd.read_csv(cs.conversations, encoding='utf-8')
    # Rename columns to match expected format (lowercase for consistency)
    df = df.rename(columns={'Utterance': 'conversation', 'Emotion': 'emotion'})
    
    return df

def train_dev_test_split(df):
    """
    Splits a DataFrame into train/dev/test sets (80/10/10).
    Returns train, dev, test DataFrames.
    """
    train, dev, test = np.split(
        df.sample(frac=1, random_state=42),  # shuffle with fixed seed
        [int(.8*len(df)), int(.9*len(df))]
    )
    return train, dev, test
