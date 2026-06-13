import pandas as pd
import matplotlib.pyplot as plt
from modeling import common

def generate_lexicon(ffn_model, embs, outfile="data/joy_sad_lexicon.csv"):
    """
    Generate a Joy/Sad lexicon using a trained FFN model.
    - Saves full lexicon with all words and scores.
    - Extracts top Joy and Sad words.
    - Produces visualizations (distribution + bar plots).
    """
    words = embs.iw  # vocabulary list from embedding object
    scores = []

    # Iterate over vocabulary and predict Joy score for each word
    for w in words:
        vec = embs.represent(w).reshape(1, -1)
        score = ffn_model.predict(vec)[0][0]
        scores.append((w, float(score)))

    # Create DataFrame with results
    df = pd.DataFrame(scores, columns=["word", "joy_score"])
    df.to_csv(outfile, index=False)
    print(f"Full lexicon saved to {outfile} with {len(df)} entries.")

    # Sort lexicon by score (descending: Joy at top)
    df_sorted = df.sort_values(by="joy_score", ascending=False)

    # Save top Joy and top Sad words to separate CSVs
    df_sorted.head(50).to_csv("data/top_joy_words.csv", index=False)
    df_sorted.tail(50).to_csv("data/top_sad_words.csv", index=False)
    print("Top Joy and Sad words saved as CSVs.")

    # Plot histogram of all scores
    plt.figure(figsize=(8, 5))
    plt.hist(df["joy_score"], bins=50, color="skyblue", edgecolor="black")
    plt.title("Distribution of Joy-Sad Scores")
    plt.xlabel("Joy Score (0=Sad, 1=Joy)")
    plt.ylabel("Word Count")
    plt.tight_layout()
    plt.savefig("data/lexicon_distribution.png")
    plt.close()

    # Plot top Joy words (horizontal bar chart)
    plt.figure(figsize=(10, 5))
    top_joy = df_sorted.head(20)
    plt.barh(top_joy["word"], top_joy["joy_score"], color="green")
    plt.gca().invert_yaxis()
    plt.title("Top 20 Joy Words")
    plt.xlabel("Joy Score")
    plt.tight_layout()
    plt.savefig("data/top_joy_words.png")
    plt.close()

    # Plot top Sad words (horizontal bar chart)
    plt.figure(figsize=(10, 5))
    top_sad = df_sorted.tail(20)
    plt.barh(top_sad["word"], top_sad["joy_score"], color="red")
    plt.gca().invert_yaxis()
    plt.title("Top 20 Sad Words")
    plt.xlabel("Joy Score")
    plt.tight_layout()
    plt.savefig("data/top_sad_words.png")
    plt.close()

    print("Visualizations saved in data/:")
    print(" - lexicon_distribution.png")
    print(" - top_joy_words.png")
    print(" - top_sad_words.png")


if __name__ == "__main__":
    # Load embeddings (limit to 50k for speed; adjust if needed)
    embs = common.get_facebook_fasttext_common_crawl(vocab_limit=50000)

    # Recreate FFN model with same structure as training
    ffn_model = common.get_ffn(
        units=[300, 256, 128, 1],
        dropout_hidden=.5,
        dropout_embedding=.2,
        learning_rate=1e-3,
        problem="classification"
    )
    # Load previously saved weights from experiment.py
    ffn_model.load_weights("results/ffn_model.weights.h5")

    # Generate lexicon and visualizations
    generate_lexicon(ffn_model, embs)
