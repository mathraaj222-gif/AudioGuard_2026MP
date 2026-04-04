import os
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ==========================================
# HYPOTHETICAL DATASET LOGITS/PROBABILITIES
# Replace these with actual inference outputs from your models
# ==========================================

# Example parameters:
NUM_SAMPLES = 1000
NUM_CLASSES = 7  # e.g., 7 emotions

def mock_get_model_predictions(model_name, num_samples, num_classes, accuracy):
    """
    Creates mock probabilities for demonstration.
    In reality, you will load the saved predictions/logits from your `outputs/` folder.
    """
    np.random.seed(hash(model_name) % (2**32 - 1))
    # True labels
    y_true = np.random.randint(0, num_classes, size=num_samples)
    
    # Mock probabilities based on accuracy
    y_pred_probs = np.random.rand(num_samples, num_classes)
    for i in range(num_samples):
        if np.random.rand() < accuracy:
            # confident in the correct class
            y_pred_probs[i, y_true[i]] += 5.0
        else:
            # confident in a wrong class
            wrong_class = (y_true[i] + 1) % num_classes
            y_pred_probs[i, wrong_class] += 5.0
            
    # softmax
    y_pred_probs = np.exp(y_pred_probs) / np.sum(np.exp(y_pred_probs), axis=1, keepdims=True)
    return y_pred_probs, y_true

# ==========================================
# 1. LOAD DATA FROM BASE MODELS
# ==========================================

# Finest Models (Front Row)
# e.g., S3_wav2vec_bert & T3_deberta_large_nli
print("Loading predictions for Finest Models (Front Row)...")
ser_best_probs, y_true = mock_get_model_predictions("S3_wav2vec_bert", NUM_SAMPLES, NUM_CLASSES, 0.92)
tca_best_probs, _      = mock_get_model_predictions("T3_deberta_large_nli", NUM_SAMPLES, NUM_CLASSES, 0.94)

# Other Best Models (Fallback Row)
# e.g., S5_wavlm_large & T4_roberta_dynabench
print("Loading predictions for Other Best Models (Fallback Row)...")
ser_other_probs, _ = mock_get_model_predictions("S5_wavlm_large", NUM_SAMPLES, NUM_CLASSES, 0.88)
tca_other_probs, _ = mock_get_model_predictions("T4_roberta_dynabench", NUM_SAMPLES, NUM_CLASSES, 0.90)

# Concatenate features (logits/probabilities) for meta-training
# X1: Front Row (Best SER + Best TCA)
X1 = np.hstack([ser_best_probs, tca_best_probs])

# X2: Fallback Row (Other Best SER + Other Best TCA)
X2 = np.hstack([ser_other_probs, tca_other_probs])

# Split into train and test sets for the meta-classifier
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    X1, X2, y_true, test_size=0.2, random_state=42
)

# ==========================================
# 2. TRAIN META CLASSIFIER 1 (FRONT ROW)
# ==========================================
print("\n--- Training Meta Classifier 1 (Front Row) ---")
# Using Random Forest as it's good at learning non-linear combinations of probabilities
meta_clf1 = RandomForestClassifier(n_estimators=100, random_state=42)
meta_clf1.fit(X1_train, y_train)

# Evaluate MC1
pred1_test = meta_clf1.predict(X1_test)
acc1 = accuracy_score(y_test, pred1_test)
print(f"Meta Classifier 1 Accuracy: {acc1:.4f}")

# ==========================================
# 3. TRAIN META CLASSIFIER 2 (FALLBACK ROW)
# ==========================================
print("\n--- Training Meta Classifier 2 (Fallback Row) ---")
meta_clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
meta_clf2.fit(X2_train, y_train)

# Evaluate MC2
pred2_test = meta_clf2.predict(X2_test)
acc2 = accuracy_score(y_test, pred2_test)
print(f"Meta Classifier 2 Accuracy: {acc2:.4f}")

# ==========================================
# 4. TWO-STAGE EVALUATION ALGORITHM
# ==========================================
print("\n--- Two-Stage Fusion Evaluation ---")

# Define confidence threshold for the first model
CONFIDENCE_THRESHOLD = 0.65

final_predictions = []
fallback_count = 0

# Get probabilities from Meta Classifier 1 for the test set
mc1_test_probs = meta_clf1.predict_proba(X1_test)
mc2_test_probs = meta_clf2.predict_proba(X2_test)

for i in range(len(y_test)):
    probs1 = mc1_test_probs[i]
    max_prob1 = np.max(probs1)
    
    if max_prob1 >= CONFIDENCE_THRESHOLD:
        # Confident enough, use First Meta Classifier
        final_preds = np.argmax(probs1)
    else:
        # Not confident, fallback to Second Meta Classifier
        fallback_count += 1
        probs2 = mc2_test_probs[i]
        final_preds = np.argmax(probs2)
        
    final_predictions.append(final_preds)

final_accuracy = accuracy_score(y_test, final_predictions)

print(f"Fallback triggered {fallback_count} times out of {len(y_test)} samples.")
print(f"Final Combined Accuracy: {final_accuracy:.4f}")
print("\nClassification Report (Two-Stage Fusion):")
print(classification_report(y_test, final_predictions))

# ==========================================
# 5. SAVE THE META CLASSIFIERS
# ==========================================
os.makedirs("outputs/meta_classifiers", exist_ok=True)
with open("outputs/meta_classifiers/meta_clf1_frontrow.pkl", "wb") as f:
    pickle.dump(meta_clf1, f)
    
with open("outputs/meta_classifiers/meta_clf2_fallback.pkl", "wb") as f:
    pickle.dump(meta_clf2, f)

print("\nModels saved to outputs/meta_classifiers/!")
