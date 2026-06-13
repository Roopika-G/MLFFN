import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
os.environ['OMP_NUM_THREADS'] = '1'       # Limit OpenMP threads
os.environ['MKL_NUM_THREADS'] = '1'       # Limit MKL threads

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Disable GPU - Force CPU-only mode
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


import pandas as pd
import os
import util
import modeling.feature_extraction as fe
from modeling import common
from sklearn.model_selection import KFold
import numpy as np
import keras
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Load conversations dataset (preprocessed)
train, dev, test = util.train_dev_test_split(util.get_conversations())
# Combine train + test for cross-validation (dev left out)
data = pd.concat([train, test], axis=0).reset_index(drop=True)

# Load embeddings (FastText common crawl)
embs = common.get_facebook_fasttext_common_crawl(vocab_limit=None)

# Define features Change data.conversations to data.text or input accordingly
FEATURES_MATRIX = fe.embedding_matrix(data.conversation, embs, common.TIMESTEPS)
FEATURES_CENTROID = fe.embedding_centroid(data.conversation, embs)

#FEATURES_MATRIX = fe.embedding_matrix(data.conversation, embs, common.TIMESTEPS)
#FEATURES_CENTROID = fe.embedding_centroid(data.conversation, embs)

# Define models (CNN, FFN, Ridge)
MODELS = {
    "cnn": lambda: common.get_cnn(
        input_shape=[common.TIMESTEPS, 300],
        num_outputs=1,
        num_filters=100,
        learning_rate=1e-3, #original
        #learning_rate=5e-4,#hypermater tuning for overfitting
        dropout_conv=.5,
        problem="classification"
    ),
    "ffn": lambda: common.get_ffn(
        units=[300, 256, 128, 1],
        dropout_hidden=.5, #original
        dropout_embedding=.2, #original
        #dropout_hidden=.8, #hypermater tuning for overfitting
        #dropout_embedding=.5, #hypermater tuning for overfitting
        learning_rate=1e-3, #original
        #learning_rate=5e-4, #hypermater tuning for overfitting
        problem="classification"
    ),
    "ridge": lambda: RidgeCV(alphas=[1, 0.5, 0.1, 0.01])
}

# Early stopping callback to avoid overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=20, verbose=1, mode="auto" #original
    #monitor="val_loss", patience=2, verbose=1, mode="auto" #hypermater tuning for overfitting
)

# Create binary labels: Joyful=1, Sad=0 for ED dataset
#data["emotion_bin"] = (data["emotion"] == "joyful").astype(int)

# Run 5-fold cross-validation
kf_iterator = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize results storage
results = {"cnn": [], "ffn": [], "ridge": []}

for i, (train_idx, test_idx) in enumerate(kf_iterator.split(data)):
    # Split labels Change data["emotion_bin" to data["labels"] or input accordingly
    #train_labels = data["emotion_bin"].iloc[train_idx]
    #test_labels = data["emotion_bin"].iloc[test_idx]
    #train_labels = data["label"].iloc[train_idx]
    #test_labels = data["label"].iloc[test_idx]
    train_labels = data["emotion"].iloc[train_idx]
    test_labels = data["emotion"].iloc[test_idx]

    # Split features
    train_features_centroid = FEATURES_CENTROID[train_idx]
    train_features_matrix = FEATURES_MATRIX[train_idx]
    test_features_centroid = FEATURES_CENTROID[test_idx]
    test_features_matrix = FEATURES_MATRIX[test_idx]

    # Loop over models
    for name, model_fun in MODELS.items():
        model = model_fun()
        if name == "cnn":
            # CNN uses embedding matrix (sequence features)
            model.fit(train_features_matrix, train_labels,
                      validation_split=.1, epochs=50, batch_size=32, #original
                      #validation_split=.3, epochs=20, batch_size=64, #hypermater tuning for overfitting
                      callbacks=[early_stopping])
            preds = model.predict(test_features_matrix)
        elif name == "ffn":
            # FFN uses centroid features (averaged embeddings)
            history = model.fit(train_features_centroid, train_labels,
                      validation_split=.1, epochs=50, batch_size=32, #original
                      #validation_split=.3, epochs=20, batch_size=64, #hypermater tuning for overfitting
                      callbacks=[early_stopping])
            # Plot training and validation loss
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss', color='blue')
            plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
            plt.title('FFN Model - Training and Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save loss plot
            os.makedirs("data", exist_ok=True)
            plt.savefig("data_ED/ffn_training_loss.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Training/Validation Loss plot saved: data_ED/ffn_training_loss.png")

            preds = model.predict(test_features_centroid)
            # Save FFN weights for lexicon generation later
            if not os.path.exists("results"):
                os.makedirs("results")
            model.save_weights("results/ffn_model.weights.h5")
            # Get predictions for validation and testing
            val_preds = model.predict(train_features_centroid[:int(len(train_features_centroid)*0.1)])
            test_preds = model.predict(test_features_centroid)
            
            # Get validation labels (first 10% of training data)
            val_labels = train_labels[:int(len(train_labels)*0.1)]
            
            # Calculate ROC curves
            val_fpr, val_tpr, _ = roc_curve(val_labels, val_preds)
            test_fpr, test_tpr, _ = roc_curve(test_labels, test_preds)
            
            val_auc = auc(val_fpr, val_tpr)
            test_auc = auc(test_fpr, test_tpr)
            
            # Plot ROC curves
            plt.figure(figsize=(10, 6))
            plt.plot(val_fpr, val_tpr, color='blue', lw=2, 
                    label=f'Validation ROC (AUC = {val_auc:.3f})')
            plt.plot(test_fpr, test_tpr, color='red', lw=2, 
                    label=f'Testing ROC (AUC = {test_auc:.3f})')
            plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('FFN Model - ROC Curves (Validation vs Testing)')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            # Save ROC curve
            os.makedirs("data", exist_ok=True)
            plt.savefig("data_ED/ffn_roc_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ ROC Curves saved: data_ED/ffn_roc_curves.png")
            print(f"   Validation AUC: {val_auc:.3f}, Testing AUC: {test_auc:.3f}")
        else:
            # Ridge regression also uses centroid features
            model.fit(train_features_centroid, train_labels)
            preds = model.predict(test_features_centroid)
            # Save Ridge model for lexicon generation
            if not os.path.exists("results"):
                os.makedirs("results")
            import joblib
            joblib.dump(model, "results/ridge_model.pkl")

        # Calculate accuracy for current fold + model
        accuracy = np.mean((preds.flatten() > 0.5) == test_labels)
        results[name].append(accuracy)
        print(f"Fold {i+1}, {name}, accuracy: {accuracy:.4f}")



# Plot 5-fold cross-validation results
plt.figure(figsize=(12, 8))

# Plot accuracy for each fold
fold_numbers = range(1, 6)
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

plt.subplot(2, 2, 1)
for i, (model_name, accuracies) in enumerate(results.items()):
    plt.plot(fold_numbers, accuracies, marker='o', linewidth=2, 
             label=f'{model_name.upper()}', color=colors[i])
plt.title('5-Fold Cross-Validation Accuracy')
plt.xlabel('Fold Number')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot box plot of accuracies
plt.subplot(2, 2, 2)
model_names = list(results.keys())
accuracies_list = list(results.values())
plt.boxplot(accuracies_list, labels=model_names)
plt.title('Accuracy Distribution Across Folds')
plt.ylabel('Accuracy')
plt.grid(True, alpha=0.3)

# Plot mean accuracy comparison
plt.subplot(2, 2, 3)
mean_accuracies = [np.mean(acc) for acc in accuracies_list]
std_accuracies = [np.std(acc) for acc in accuracies_list]
bars = plt.bar(model_names, mean_accuracies, yerr=std_accuracies, 
               color=colors[:len(model_names)], alpha=0.7, capsize=5)
plt.title('Mean Accuracy Comparison')
plt.ylabel('Mean Accuracy')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Plot accuracy heatmap
plt.subplot(2, 2, 4)
accuracy_matrix = np.array(accuracies_list).T
im = plt.imshow(accuracy_matrix, cmap='YlOrRd', aspect='auto')
plt.colorbar(im, label='Accuracy')
plt.title('Accuracy Heatmap (Folds × Models)')
plt.xlabel('Models')
plt.ylabel('Folds')
plt.xticks(range(len(model_names)), model_names)
plt.yticks(range(5), [f'Fold {i+1}' for i in range(5)])

plt.tight_layout()
plt.savefig("data_ED/ffn_cross_validation_results.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Cross-validation results plot saved: data_ED/ffn_cross_validation_results.png")

# Save results to TSV files for ttest analysis
if not os.path.exists("results"):
    os.makedirs("results")

for model_name, accuracies in results.items():
    df_results = pd.DataFrame({"fold": range(1, 6), "accuracy": accuracies})
    df_results.to_csv(f"results/{model_name}.tsv", sep='\t', index=False)
    print(f"✅ Saved {model_name} results to results/{model_name}.tsv")

# Generate testing results summary
plt.figure(figsize=(12, 6))

# Plot final testing results
plt.subplot(1, 2, 1)
model_names = list(results.keys())
mean_accuracies = [np.mean(acc) for acc in results.values()]
std_accuracies = [np.std(acc) for acc in results.values()]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

bars = plt.bar(model_names, mean_accuracies, yerr=std_accuracies, 
               color=colors, alpha=0.7, capsize=5)
plt.title('Final Testing Results (5-Fold CV)')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)

# Add value labels on bars
for i, (bar, mean_acc, std_acc) in enumerate(zip(bars, mean_accuracies, std_accuracies)):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std_acc + 0.01,
             f'{mean_acc:.3f}±{std_acc:.3f}', ha='center', va='bottom')

# Plot model performance comparison
plt.subplot(1, 2, 2)
best_model = model_names[np.argmax(mean_accuracies)]
best_accuracy = max(mean_accuracies)

plt.pie([best_accuracy, 1-best_accuracy], 
        labels=[f'{best_model.upper()}\n({best_accuracy:.3f})', 'Others'],
        colors=['#FF6B6B', '#E0E0E0'], autopct='%1.1f%%')
plt.title(f'Best Performing Model: {best_model.upper()}')

plt.tight_layout()
plt.savefig("data_ED/ffn_testing_results_summary.png", dpi=300, bbox_inches='tight')
plt.close()

print("✅ Testing results summary saved: data_ED/ffn_testing_results_summary.png")
print(f"✅ Best model: {best_model.upper()} with accuracy: {best_accuracy:.3f}")

print("\n📊 Cross-validation summary:")
for model_name, accuracies in results.items():
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"{model_name.upper()}: {mean_acc:.4f} ± {std_acc:.4f}")

# Cleanup TensorFlow session
import gc
tf.keras.backend.clear_session()
gc.collect()
print("\n🧹 Cleaned up TensorFlow session")
