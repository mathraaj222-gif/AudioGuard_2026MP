import numpy as np

def soft_voting_fusion(tca_probs, ser_probs, tca_weight=0.6, ser_weight=0.4):
    """
    Weighted Average Fusion (Soft Voting)
    Requires NO external data or training.
    
    tca_probs: Array of probabilities from the Text Context Analysis model (e.g. 7 classes)
    ser_probs: Array of probabilities from the Speech Emotion Recognition model (e.g. 7 classes)
    """
    # Simply weight and add the probabilities
    merged_probs = (tca_probs * tca_weight) + (ser_probs * ser_weight)
    
    # The final prediction is the class with the highest merged probability
    final_pred = np.argmax(merged_probs)
    return final_pred, merged_probs

def max_confidence_fusion(tca_probs, ser_probs):
    """
    Max Confidence Fusion
    Whichever model is more confident dictates the final output.
    """
    tca_max = np.max(tca_probs)
    ser_max = np.max(ser_probs)
    
    if tca_max > ser_max:
        return np.argmax(tca_probs), tca_probs
    else:
        return np.argmax(ser_probs), ser_probs

def fallback_fusion(tca_probs, ser_probs, tca_threshold=0.75):
    """
    Rule-based Fallback Fusion
    Prioritize the Text (TCA) model because text is often a stronger indicator for hate speech.
    If the TCA model is unsure (< threshold), fallback to the Audio (SER) model.
    """
    tca_max = np.max(tca_probs)
    if tca_max >= tca_threshold:
        return np.argmax(tca_probs), tca_probs
    else:
        return np.argmax(ser_probs), ser_probs

# ==========================================
# EXAMPLE USAGE
# ==========================================
if __name__ == "__main__":
    # Example probabilities for a single sample (e.g. 7 emotion/hate speech classes)
    # Suppose Class 1 = "Hate Speech / Anger", Class 0 = "Neutral", etc.
    mock_tca_probs = np.array([0.2, 0.45, 0.05, 0.1, 0.05, 0.05, 0.1])
    mock_ser_probs = np.array([0.1, 0.80, 0.0,  0.0, 0.0,  0.0,  0.1])
    
    print("TCA Prediction:", np.argmax(mock_tca_probs), "| Confidence:", np.max(mock_tca_probs))
    print("SER Prediction:", np.argmax(mock_ser_probs), "| Confidence:", np.max(mock_ser_probs))
    print("-" * 40)
    
    # 1. Soft Voting
    pred_soft, probs_soft = soft_voting_fusion(mock_tca_probs, mock_ser_probs, tca_weight=0.6, ser_weight=0.4)
    print(f"Soft Voting Fusion Output : Class {pred_soft} (Probs: {np.round(probs_soft, 2)})")
    
    # 2. Max Confidence
    pred_max, probs_max = max_confidence_fusion(mock_tca_probs, mock_ser_probs)
    print(f"Max Confidence Fusion     : Class {pred_max} (Probs: {np.round(probs_max, 2)})")
    
    # 3. Fallback Pipeline
    pred_fall, probs_fall = fallback_fusion(mock_tca_probs, mock_ser_probs, tca_threshold=0.7)
    print(f"Fallback Fusion           : Class {pred_fall} (Probs: {np.round(probs_fall, 2)})")

