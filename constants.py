# constants.py
# Holds file paths for dataset and embeddings. Update ENV variables for your environment.

import os
from os.path import join as jn

# Root of project (default current directory)
root = os.environ.get("EMPATHY_PROJECT_ROOT", ".")

# Path to preprocessed conversations dataset (created by preprocess_dialogues.py)
#conversations = jn(root, "data", "empathetic_dialogues.csv")
#conversations = jn(root, "data", "IEmpathize_Dataset.csv")
#conversations = jn(root, "data", "dair_ai_emotion_joy_sadness.csv")
#conversations = jn(root, "data", "dailydialog_happiness_sadness.csv")
#conversations = jn(root, "data", "meld_joy_sadness_only.csv")
#conversations = jn(root, "data_GoEmotions", "go_emotion_joy_sadness_mutually_exclusive.csv")
conversations = jn(root, "data_ED", "empathetic_dialogues.csv")

# Embedding paths (set VECTORS env variable to your embeddings folder)
vectors = os.environ.get("VECTORS", "./embeddings")
facebook_fasttext_common_crawl = jn(vectors, "crawl-300d-2M.vec")
google_news_embeddings = jn(vectors, "GoogleNews-vectors-negative300.bin")
