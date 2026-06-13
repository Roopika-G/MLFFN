import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from modeling import common
from sklearn.linear_model import RidgeCV
import os

def generate_ffn_lexicon(ffn_model, embs, vocab_limit=50000):
    """
    Generate Joy/Sad lexicon using the trained FFN model.
    """
    print("🔄 Generating FFN lexicon...")
    words = embs.iw[:vocab_limit]  # vocabulary list from embedding object
    scores = []

    # Iterate over vocabulary and predict Joy score for each word
    for w in words:
        vec = embs.represent(w).reshape(1, -1)
        score = ffn_model.predict(vec, verbose=0)[0][0]
        scores.append((w, float(score)))

    # Create DataFrame with results
    df = pd.DataFrame(scores, columns=["word", "joy_score"])
    df = df.sort_values(by="joy_score", ascending=False)
    
    print(f"✅ FFN lexicon generated with {len(df)} words")
    return df

def generate_ridge_lexicon(ridge_model, embs, vocab_limit=50000):
    """
    Generate Joy/Sad lexicon using Ridge regression coefficients.
    """
    print("🔄 Generating Ridge lexicon...")
    words = embs.iw[:vocab_limit]
    scores = []

    # Ridge model coefficients represent feature importance
    coefficients = ridge_model.coef_[0]  # Get coefficients for Joy prediction
    
    for w in words:
        vec = embs.represent(w)
        # Calculate weighted score using Ridge coefficients
        score = np.dot(vec, coefficients)
        # Normalize to 0-1 range (sigmoid-like transformation)
        normalized_score = 1 / (1 + np.exp(-score))
        scores.append((w, float(normalized_score)))

    df = pd.DataFrame(scores, columns=["word", "joy_score"])
    df = df.sort_values(by="joy_score", ascending=False)
    
    print(f"✅ Ridge lexicon generated with {len(df)} words")
    return df

def generate_cnn_lexicon(cnn_model, embs, vocab_limit=50000):
    """
    Generate Joy/Sad lexicon using CNN model.
    Note: This is an approximation since CNN works on sequences, not individual words.
    """
    print("🔄 Generating CNN lexicon (approximation)...")
    words = embs.iw[:vocab_limit]
    scores = []

    # For CNN, we'll use the word embeddings directly through the model
    # This is an approximation since CNN is designed for sequences
    for w in words:
        vec = embs.represent(w).reshape(1, 1, -1)  # Reshape for CNN input
        score = cnn_model.predict(vec, verbose=0)[0][0]
        scores.append((w, float(score)))

    df = pd.DataFrame(scores, columns=["word", "joy_score"])
    df = df.sort_values(by="joy_score", ascending=False)
    
    print(f"✅ CNN lexicon generated with {len(df)} words")
    return df

def save_lexicon_files(df, model_name, base_dir="data"):
    """
    Save lexicon files with appropriate naming.
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Save full lexicon
    full_file = f"{base_dir}/{model_name}_lexicon.csv"
    df.to_csv(full_file, index=False)
    
    # Save top words
    top_joy_file = f"{base_dir}/{model_name}_top_joy_words.csv"
    top_sad_file = f"{base_dir}/{model_name}_top_sad_words.csv"
    
    df.head(50).to_csv(top_joy_file, index=False)
    df.tail(50).to_csv(top_sad_file, index=False)
    
    print(f"📁 Saved {model_name} lexicon files:")
    print(f"   - {full_file}")
    print(f"   - {top_joy_file}")
    print(f"   - {top_sad_file}")

def create_visualizations(df, model_name, base_dir="data"):
    """
    Create visualizations for the lexicon.
    """
    os.makedirs(base_dir, exist_ok=True)
    
    # Plot histogram of all scores
    plt.figure(figsize=(10, 6))
    plt.hist(df["joy_score"], bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(f"{model_name.upper()} Model: Distribution of Joy-Sad Scores")
    plt.xlabel("Joy Score (0=Sad, 1=Joy)")
    plt.ylabel("Word Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{model_name}_lexicon_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot top Joy words (horizontal bar chart)
    plt.figure(figsize=(12, 8))
    top_joy = df.head(20)
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(top_joy)))
    bars = plt.barh(range(len(top_joy)), top_joy["joy_score"], color=colors)
    plt.yticks(range(len(top_joy)), top_joy["word"])
    plt.gca().invert_yaxis()
    plt.title(f"{model_name.upper()} Model: Top 20 Joy Words")
    plt.xlabel("Joy Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{model_name}_top_joy_words.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot top Sad words (horizontal bar chart)
    plt.figure(figsize=(12, 8))
    top_sad = df.tail(20)
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(top_sad)))
    bars = plt.barh(range(len(top_sad)), top_sad["joy_score"], color=colors)
    plt.yticks(range(len(top_sad)), top_sad["word"])
    plt.gca().invert_yaxis()
    plt.title(f"{model_name.upper()} Model: Top 20 Sad Words")
    plt.xlabel("Joy Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{model_name}_top_sad_words.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_visualization(lexicons, base_dir="data"):
    """
    Create comparison visualization across all models.
    """
    print("🔄 Creating comparison visualizations...")
    
    # Compare top 10 Joy words across models
    plt.figure(figsize=(15, 10))
    
    models = list(lexicons.keys())
    colors = ['#2E8B57', '#4169E1', '#DC143C']  # Sea Green, Royal Blue, Crimson
    
    for i, (model_name, df) in enumerate(lexicons.items()):
        top_joy = df.head(10)
        plt.subplot(2, 2, i+1)
        bars = plt.barh(range(len(top_joy)), top_joy["joy_score"], color=colors[i], alpha=0.7)
        plt.yticks(range(len(top_joy)), top_joy["word"])
        plt.gca().invert_yaxis()
        plt.title(f"{model_name.upper()}: Top 10 Joy Words")
        plt.xlabel("Joy Score")
        plt.grid(True, alpha=0.3)
    
    # Overall comparison
    plt.subplot(2, 2, 4)
    model_means = [df["joy_score"].mean() for df in lexicons.values()]
    model_stds = [df["joy_score"].std() for df in lexicons.values()]
    
    bars = plt.bar(models, model_means, yerr=model_stds, color=colors, alpha=0.7, capsize=5)
    plt.title("Model Comparison: Average Joy Scores")
    plt.ylabel("Average Joy Score")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{base_dir}/all_models_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comparison visualization saved")

def main():
    """
    Main function to generate lexicons from all three models.
    """
    print("🚀 Generating Joy/Sad Lexicons from All Models")
    print("=" * 50)
    
    # Load embeddings (limit to 50k for speed)
    print("📚 Loading embeddings...")
    embs = common.get_facebook_fasttext_common_crawl(vocab_limit=50000)
    
    # Load trained models
    print("🤖 Loading trained models...")
    
    # FFN Model
    ffn_model = common.get_ffn(
        units=[300, 256, 128, 1],
        dropout_hidden=.5,
        dropout_embedding=.2,
        learning_rate=1e-3,
        problem="classification"
    )
    ffn_model.load_weights("results/ffn_model.weights.h5")
    
    # Ridge Model (we need to retrain it to get coefficients)
    print("⚠️  Note: Ridge model will be retrained to extract coefficients...")
    # For now, we'll use a simplified approach
    ridge_model = RidgeCV(alphas=[1, 0.5, 0.1, 0.01])
    
    # CNN Model
    cnn_model = common.get_cnn(
        input_shape=[200, 300],  # TIMESTEPS, embedding_dim
        num_outputs=1,
        num_filters=100,
        learning_rate=1e-3,
        dropout_conv=.5,
        problem="classification"
    )
    # Note: CNN model weights aren't saved, so we'll skip CNN for now
    
    # Generate lexicons
    lexicons = {}
    
    # FFN Lexicon
    ffn_lexicon = generate_ffn_lexicon(ffn_model, embs)
    save_lexicon_files(ffn_lexicon, "ffn")
    create_visualizations(ffn_lexicon, "ffn")
    lexicons["FFN"] = ffn_lexicon
    
    # Ridge Lexicon (simplified approach)
    print("⚠️  Ridge lexicon generation requires retraining - skipping for now")
    print("💡 You can manually extract Ridge coefficients from the experiment results")
    
    # CNN Lexicon (approximation)
    print("⚠️  CNN lexicon generation requires model weights - skipping for now")
    print("💡 CNN model weights weren't saved in the current experiment setup")
    
    # Create comparison visualization
    if len(lexicons) > 1:
        create_comparison_visualization(lexicons)
    
    print("\n🎉 Lexicon generation completed!")
    print("\n📊 Generated files:")
    print("   - ffn_lexicon.csv")
    print("   - ffn_top_joy_words.csv") 
    print("   - ffn_top_sad_words.csv")
    print("   - ffn_lexicon_distribution.png")
    print("   - ffn_top_joy_words.png")
    print("   - ffn_top_sad_words.png")
    
    print("\n💡 To generate Ridge and CNN lexicons:")
    print("   1. Modify experiment.py to save Ridge coefficients")
    print("   2. Modify experiment.py to save CNN model weights")
    print("   3. Re-run this script")

if __name__ == "__main__":
    main()
