# Empathetic FFN - Adapted from Empathic Reactions

This project adapts the Mixed-Level Feed Forward Network (MLFFN) pipeline from the original *Empathic Reactions* repo to work with the **EmpatheticDialogues** dataset (HuggingFace), using only conversations labeled **Joy** and **Sad**.

---

## 📂 Project Structure

```
empathetic_ffn/
│
├── preprocess_dialogues.py    # Prepares dataset: filters Joy/Sad, groups conversations, saves CSV
├── constants.py                # Paths for data and embeddings
├── util.py                     # Data loading and splitting helpers
├── experiment.py               # Main training/evaluation loop (FFN, CNN, Ridge)
├── run_experiments.sh          # Shell script to run preprocessing + training + testing
├── README.md                   # This file
│
├── data/
│   └── empathetic_dialogues.csv (generated after running preprocess)
│
└── modeling/
    ├── common.py               # Model definitions: FFN, CNN, RNN, etc.
    ├── feature_extraction.py   # Converts text to embedding matrices/centroids
    ├── embedding.py            # Loads word embeddings (FastText, Word2Vec)
    ├── evaluation.py           # Cross-validation and evaluation utilities
    └── ttest.py                # Statistical tests between model results
```

---

## ⚙️ Setup

1. Clone or extract this repo.
2. Install dependencies:
   ```bash
   pip install datasets pandas numpy scikit-learn keras tensorflow gensim
   ```
3. Download embeddings (e.g., FastText `crawl-300d-2M.vec.zip`) and set path:
   ```bash
   export VECTORS=/path/to/embeddings
   export EMPATHY_PROJECT_ROOT=.
   ```

---

## ▶️ Running the Pipeline

1. Preprocess dataset and run experiments:
   ```bash
   bash run_experiments.sh
   ```

   This will:
   - Download **EmpatheticDialogues** from HuggingFace.
   - Filter conversations with `joy` and `sad` labels.
   - Train/evaluate models with cross-validation.
   - Run paired t-tests on results.

2. Check outputs:
   - `data/empathetic_dialogues.csv` → processed dataset.
   - `results/*.tsv` → per-model CV results.
   - `ttest_results.txt` → statistical test results.

---

## 📊 Models Implemented

- **FFN (MLFFN)**: Mixed-Level Feed Forward Network with dropout and dense layers.  
- **CNN**: Convolutional model over word embeddings.  
- **Ridge Regression**: Baseline linear model with embeddings.  

---

## 🔮 Notes

- By default, labels are binarized: `Joy=1`, `Sad=0`.  
- Cross-validation folds = 5 (can be changed in `experiment.py`).  
- Results are Pearson correlation (for regression) or accuracy (for classification).  

---

## 📖 Generating Joy/Sad Lexicon

1. After running experiments, ensure `results/ffn_model.h5` is saved.
2. Run:
   ```bash
   python generate_lexicon.py
   ```
3. This will create:
   - `data/joy_sad_lexicon.csv` → each word with its Joy score (closer to 1 = Joy, closer to 0 = Sad).

## 🐍 **Complete Virtual Environment Setup Guide**

I've created the necessary files for you. Here are the step-by-step instructions:

### **Method 1: Automated Setup (Recommended)**

```bash
# Navigate to the project directory
cd /home/bet68/Documents/Lexicon_Generation/MLFFN-Empathy

# Run the automated setup script
bash setup_environment.sh
```

This script will:
- Create a virtual environment named `mlffn_env`
- Install all required packages
- Create necessary directories
- Provide you with activation instructions

### **Method 2: Manual Setup**

If you prefer to do it manually:

```bash
# Navigate to the project directory
cd /home/bet68/Documents/Lexicon_Generation/MLFFN-Empathy

# Create virtual environment
python3 -m venv mlffn_env

# Activate the virtual environment
source mlffn_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install all requirements
pip install -r requirements.txt

# Create necessary directories
mkdir -p data results embeddings
```

### **3. Activate the Virtual Environment**

Every time you want to work on this project:

```bash
cd /home/bet68/Documents/Lexicon_Generation/MLFFN-Empathy
source mlffn_env/bin/activate
```

You'll see `(mlffn_env)` at the beginning of your terminal prompt, indicating the environment is active.

### **4. Download Word Embeddings**

You'll need word embeddings. Here are the options:

**Option A: FastText Common Crawl (Recommended)**
```bash
# Create embeddings directory
mkdir -p embeddings
cd embeddings

# Download FastText embeddings (~7GB)
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip

# Go back to project root
cd ..
```

**Option B: Google News Embeddings**
```bash
# Download Google News embeddings (~3.4GB)
# Note: You'll need to download this manually from Google Drive
# Place it in the embeddings/ directory
```

### **5. Set Environment Variables**

```bash
# Set the path to your embeddings
export VECTORS=/home/bet68/Documents/Lexicon_Generation/MLFFN-Empathy/embeddings

# Set project root (optional)
export EMPATHY_PROJECT_ROOT=/home/bet68/Documents/Lexicon_Generation/MLFFN-Empathy
```

### **6. Run the Project**

```bash
# Make sure you're in the virtual environment
source mlffn_env/bin/activate

# Run the complete pipeline
bash run_experiments.sh

# Generate the lexicon
python generate_lexicon.py
```

### **7. Deactivate When Done**

```bash
deactivate
```

### **8. Quick Verification**

To verify everything is set up correctly:

```bash
# Activate environment
source mlffn_env/bin/activate

# Check Python packages
pip list

# Test imports
python -c "import tensorflow, keras, pandas, numpy, sklearn, gensim, datasets, matplotlib; print('✅ All packages imported successfully!')"
```

### **9. Troubleshooting**

If you encounter issues:

**Python version issues:**
```bash
# Check Python version (should be 3.8+)
python3 --version

# If needed, install python3-venv
sudo apt-get install python3-venv
```

**Package conflicts:**
```bash
# If you have conflicts, recreate the environment
rm -rf mlffn_env
bash setup_environment.sh
```

**Memory issues with embeddings:**
```bash
# If you have limited RAM, use a smaller vocabulary limit
# Edit generate_lexicon.py line 74: vocab_limit=10000
```


[1 tool called]

## **Summary**

I've created:
1. **`requirements.txt`** - Lists all required Python packages
2. **`setup_environment.sh`** - Automated setup script
3. **Complete step-by-step guide** above

**Quick Start:**
```bash
cd /home/bet68/Documents/Lexicon_Generation/MLFFN-Empathy
bash setup_environment.sh
source mlffn_env/bin/activate
# Download embeddings, set environment variables, then run experiments
```
## 🎯 **Enhanced Lexicon Generation System**

### **Files Created:**

1. **`generate_all_lexicons.py`** - Comprehensive version (for future use)
2. **`generate_lexicons_simple.py`** - Working version for FFN + Ridge
3. **Updated `experiment.py`** - Now saves Ridge model
4. **Updated `requirements.txt`** - Added joblib dependency

### **What the New System Does:**

**🤖 Generates Lexicons from Multiple Models:**
- **FFN Model** - Uses neural network predictions
- **Ridge Model** - Uses linear coefficients (feature importance)
- **CNN Model** - Ready for future implementation

**📁 Proper File Organization:**
```
data/
├── ffn_lexicon.csv              # Full FFN lexicon
├── ffn_top_joy_words.csv        # Top 50 Joy words (FFN)
├── ffn_top_sad_words.csv        # Top 50 Sad words (FFN)
├── ridge_lexicon.csv            # Full Ridge lexicon  
├── ridge_top_joy_words.csv      # Top 50 Joy words (Ridge)
├── ridge_top_sad_words.csv      # Top 50 Sad words (Ridge)
├── ffn_lexicon_distribution.png # FFN score distribution
├── ffn_top_joy_words.png        # FFN top Joy visualization
├── ffn_top_sad_words.png        # FFN top Sad visualization
├── ridge_lexicon_distribution.png # Ridge score distribution
├── ridge_top_joy_words.png      # Ridge top Joy visualization
├── ridge_top_sad_words.png      # Ridge top Sad visualization
└── models_comparison.png        # Cross-model comparison
```

**📊 Rich Visualizations:**
- **Distribution plots** - Show how Joy scores are distributed
- **Top words charts** - Horizontal bar charts of top Joy/Sad words
- **Model comparison** - Side-by-side comparison of models
- **Overlap analysis** - Shows which words both models agree on

### **How to Use:**

**Step 1: Run experiments (if not done already)**
```bash
bash run_experiments.sh
```

**Step 2: Generate lexicons from all models**
```bash
python generate_lexicons_simple.py
```

### **Key Features:**

**🎨 Beautiful Visualizations:**
- Color-coded charts (Green for Joy, Red for Sad)
- High-resolution PNG files (300 DPI)
- Professional styling with grids and proper labels

**📈 Model Comparison:**
- Side-by-side comparison of top words
- Statistical analysis of model differences
- Word overlap analysis between models

**⚡ Optimized Performance:**
- Processes 10,000 words (adjustable)
- Progress indicators during processing
- Memory-efficient implementation

**🔬 Research-Ready:**
- Proper statistical normalization
- Comprehensive file naming
- Detailed logging and progress tracking

The system will generate lexicons from both your best-performing models (FFN and Ridge) with beautiful visualizations and proper file organization. This gives you a complete emotion lexicon system that you can use for research or applications!

mkdir -p data
bash run_experiments.sh > data/experiment_results.txt 2>&1
