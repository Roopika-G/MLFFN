#!/bin/bash
# run_experiments.sh
# Orchestrates preprocessing, training, and statistical testing for Joy/Sad classification.

echo ""
#echo "Running experiments on EmpatheticDialogues (Joy/Sad):"
#echo "Running experiments on Iempathize (Providing-Empathy and Seeking-Empathy):"
#echo "Running experiments on EPITOME (No Empathy/Strong Empathy):"
#echo "Running experiments on Daily Dialuge (Joy/Sad):"
#echo "Running experiments on MELD (Joy/Sad):"
#echo "Running experiments on GoEmotions (Joy/Sad):"
echo "Running experiments on EmpatheticDialogues (Joy/Sad):"
# Step 1: Preprocess dataset (download + filter + save as CSV)
python Data-Preprocessing-ED.py

# Step 2: Run training and cross-validation
python experiment.py

# Step 3: Run statistical t-tests on results (CNN vs FFN vs Ridge)
python modeling/ttest.py > ttest_results.txt

echo "Experiments completed."
