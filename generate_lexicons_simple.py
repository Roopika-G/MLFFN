import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from modeling import common
import joblib
import os

def generate_ffn_lexicon(ffn_model, embs, vocab_limit=10000):
    """
    Generate Joy/Sad lexicon using the trained FFN model.
    """
    print("🔄 Generating FFN lexicon...")
    words = embs.iw[:vocab_limit]
    scores = []

    for i, w in enumerate(words):
        if i % 1000 == 0:
            print(f"   Processing word {i}/{vocab_limit}")
        vec = embs.represent(w).reshape(1, -1)
        score = ffn_model.predict(vec, verbose=0)[0][0]
        scores.append((w, float(score)))

    df = pd.DataFrame(scores, columns=["word", "Joy_Score"])
    df = df.sort_values(by="Joy_Score", ascending=False)
    
    print(f"✅ FFN lexicon generated with {len(df)} words")
    return df

def generate_ridge_lexicon(ridge_model, embs, vocab_limit=10000):
    """
    Generate Joy/Sad lexicon using Ridge regression coefficients.
    """
    print("🔄 Generating Ridge lexicon...")
    words = embs.iw[:vocab_limit]
    scores = []

    # Ridge model coefficients represent feature importance
    coefficients = ridge_model.coef_[0]
    
    for i, w in enumerate(words):
        if i % 1000 == 0:
            print(f"   Processing word {i}/{vocab_limit}")
        vec = embs.represent(w)
        # Calculate weighted score using Ridge coefficients
        score = np.dot(vec, coefficients)
        # Ensure score is a scalar
        if np.isscalar(score):
            score_value = score
        else:
            score_value = score[0] if len(score) > 0 else 0.0
        # Normalize to 0-1 range (sigmoid-like transformation)
        normalized_score = 1 / (1 + np.exp(-score_value))
        scores.append((w, float(normalized_score)))

    df = pd.DataFrame(scores, columns=["word", "Joy_Score"])
    df = df.sort_values(by="Joy_Score", ascending=False)
    
    print(f"✅ Ridge lexicon generated with {len(df)} words")
    return df

def save_lexicon_files(df, model_name, base_dir="data_ED"):
    """Save lexicon files with appropriate naming."""
    os.makedirs(base_dir, exist_ok=True)
    
    # Save full lexicon
    full_file = f"{base_dir}/{model_name}_lexicon.csv"
    df.to_csv(full_file, index=False)
    
    # Save top words
    top_Joy_file = f"{base_dir}/{model_name}_top_Joy_words.csv"
    top_Sad_file = f"{base_dir}/{model_name}_top_Sad_words.csv"
    
    df.head(50).to_csv(top_Joy_file, index=False)
    df.tail(50).to_csv(top_Sad_file, index=False)
    
    print(f"📁 Saved {model_name} lexicon files:")
    print(f"   - {full_file}")
    print(f"   - {top_Joy_file}")
    print(f"   - {top_Sad_file}")

def create_visualizations(df, model_name, base_dir="data_ED"):
    """Create visualizations for the lexicon."""
    os.makedirs(base_dir, exist_ok=True)
    
    # Plot histogram of all scores
    plt.figure(figsize=(10, 6))
    plt.hist(df["Joy_Score"], bins=50, color="skyblue", edgecolor="black", alpha=0.7)
    plt.title(f"{model_name.upper()} Model: Distribution of Joy-Sad Scores")
    plt.xlabel("Joy Score (0=Sad, 1=Joy)")
    plt.ylabel("Word Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{model_name}_lexicon_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot top Joy words (horizontal bar chart)
    plt.figure(figsize=(12, 8))
    top_Joy = df.head(20)
    colors = plt.cm.Greens(np.linspace(0.4, 0.8, len(top_Joy)))
    plt.barh(range(len(top_Joy)), top_Joy["Joy_Score"], color=colors)
    plt.yticks(range(len(top_Joy)), top_Joy["word"])
    plt.gca().invert_yaxis()
    plt.title(f"{model_name.upper()} Model: Top 20 Joy Words")
    plt.xlabel("Joy Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{model_name}_top_Joy_words.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Plot top Sad words (horizontal bar chart)
    plt.figure(figsize=(12, 8))
    top_Sad = df.tail(20)
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(top_Sad)))
    plt.barh(range(len(top_Sad)), top_Sad["Joy_Score"], color=colors)
    plt.yticks(range(len(top_Sad)), top_Sad["word"])
    plt.gca().invert_yaxis()
    plt.title(f"{model_name.upper()} Model: Top 20 Sad Words")
    plt.xlabel("Joy Score")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{base_dir}/{model_name}_top_Sad_words.png", dpi=300, bbox_inches='tight')
    plt.close()

def create_comparison_visualization(lexicons, base_dir="data_ED"):
    """Create comparison visualization across models."""
    print("🔄 Creating comparison visualizations...")
    
    plt.figure(figsize=(15, 10))
    
    models = list(lexicons.keys())
    colors = ['#2E8B57', '#4169E1']  # Sea Green, Royal Blue
    
    # Compare top 10 Joy words across models
    for i, (model_name, df) in enumerate(lexicons.items()):
        top_Joy = df.head(10)
        plt.subplot(2, 2, i+1)
        bars = plt.barh(range(len(top_Joy)), top_Joy["Joy_Score"], color=colors[i], alpha=0.7)
        plt.yticks(range(len(top_Joy)), top_Joy["word"])
        plt.gca().invert_yaxis()
        plt.title(f"{model_name.upper()}: Top 10 Joy Words")
        plt.xlabel("Joy Score")
        plt.grid(True, alpha=0.3)
    
    # Overall comparison
    plt.subplot(2, 2, 3)
    model_means = [df["Joy_Score"].mean() for df in lexicons.values()]
    model_stds = [df["Joy_Score"].std() for df in lexicons.values()]
    
    bars = plt.bar(models, model_means, yerr=model_stds, color=colors, alpha=0.7, capsize=5)
    plt.title("Model Comparison: Average Joy Scores")
    plt.ylabel("Average Joy Score")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Word overlap analysis
    plt.subplot(2, 2, 4)
    if len(lexicons) == 2:
        model_names = list(lexicons.keys())
        df1, df2 = lexicons[model_names[0]], lexicons[model_names[1]]
        
        # Top 100 words from each model
        top1 = set(df1.head(100)["word"])
        top2 = set(df2.head(100)["word"])
        
        overlap = len(top1.intersection(top2))
        total_unique = len(top1.union(top2))
        
        plt.pie([overlap, total_unique - overlap], 
                labels=[f'Overlap ({overlap})', f'Unique ({total_unique - overlap})'],
                colors=['#FF6B6B', '#4ECDC4'], autopct='%1.1f%%')
        plt.title(f"Top 100 Joy Words Overlap\n{model_names[0]} vs {model_names[1]}")
    
    plt.tight_layout()
    plt.savefig(f"{base_dir}/models_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comparison visualization saved")

def main():
    """Main function to generate lexicons from FFN and Ridge models."""
    print("🚀 Generating Joy/Sad Lexicons from FFN and Ridge Models")
    print("=" * 60)
    
    # Check if models exist
    if not os.path.exists("results/ffn_model.weights.h5"):
        print("❌ FFN model weights not found. Please run experiments first.")
        return
    
    if not os.path.exists("results/ridge_model.pkl"):
        print("❌ Ridge model not found. Please run experiments first.")
        return
    
    # Load embeddings (limit to 10k for speed)
    print("📚 Loading embeddings...")
    embs = common.get_facebook_fasttext_common_crawl(vocab_limit=10000)
    
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
    
    # Ridge Model
    ridge_model = joblib.load("results/ridge_model.pkl")
    
    # Generate lexicons
    lexicons = {}
    
    # FFN Lexicon
    ffn_lexicon = generate_ffn_lexicon(ffn_model, embs)
    save_lexicon_files(ffn_lexicon, "ffn")
    create_visualizations(ffn_lexicon, "ffn")
    lexicons["FFN"] = ffn_lexicon
    
    # Ridge Lexicon
    ridge_lexicon = generate_ridge_lexicon(ridge_model, embs)
    save_lexicon_files(ridge_lexicon, "ridge")
    create_visualizations(ridge_lexicon, "ridge")
    lexicons["Ridge"] = ridge_lexicon
    
    # Create comparison visualization
    create_comparison_visualization(lexicons)
    
    print("\n🎉 Lexicon generation completed!")
    print("\n📊 Generated files:")
    for model in ["ffn", "ridge"]:
        print(f"\n{model.upper()} Model:")
        print(f"   - {model}_lexicon.csv")
        print(f"   - {model}_top_Joy_words.csv") 
        print(f"   - {model}_top_Sad_words.csv")
        print(f"   - {model}_lexicon_distribution.png")
        print(f"   - {model}_top_Joy_words.png")
        print(f"   - {model}_top_Sad_words.png")
    
    print(f"\n📈 Comparison:")
    print(f"   - models_comparison.png")
    
    print(f"\n💡 Insights:")
    print(f"   - FFN average Joy score: {ffn_lexicon['Joy_Score'].mean():.4f}")
    print(f"   - Ridge average Joy score: {ridge_lexicon['Joy_Score'].mean():.4f}")

if __name__ == "__main__":
    main()
