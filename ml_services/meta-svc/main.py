import os
import torch
import torch.nn as nn
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

# Define the Architecture (Must match your training code)
class CrossModalAttentionModel(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12):
        super().__init__()
        self.audio_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, 256), nn.ReLU(),
            nn.Linear(256, 1), nn.Sigmoid()
        )
    def forward(self, audio, text):
        a = self.audio_norm(audio).unsqueeze(1)
        t = self.text_norm(text).unsqueeze(1)
        attn_out, _ = self.cross_attn(a, t, t)
        return self.classifier(torch.cat((attn_out.squeeze(1), text), dim=1))

app = FastAPI(title="AudioGuard Meta-Classification Service")

# Load models into memory on startup
# Using try-except to handle cases where models might be missing during initial build
lgb_model = None
attn_model = None

@app.on_event("startup")
async def startup_event():
    global lgb_model, attn_model
    try:
        print("Meta-Svc: Loading Ensemble components...")
        lgb_model = joblib.load("lgb_component.pkl")
        attn_model = CrossModalAttentionModel()
        attn_model.load_state_dict(torch.load("best_attention_model.pt", map_location="cpu"))
        attn_model.eval()
        print("Meta-Svc: Models Loaded Successfully!")
    except Exception as e:
        print(f"Meta-Svc Startup ERROR: {e}")

class FusionRequest(BaseModel):
    embedding: List[float] # This expects the 1536-dim vector (768 Audio + 768 Text)

@app.get("/")
def health_check():
    return {"status": "online", "service": "meta-classifier"}

@app.post("/predict")
async def predict(data: FusionRequest):
    if lgb_model is None or attn_model is None:
        return {"error": "Models not loaded on server"}
        
    vec = np.array(data.embedding).reshape(1, -1)
    
    # 1. LightGBM Probability
    lgb_p = lgb_model.predict_proba(vec)[:, 1][0]
    
    # 2. Attention Probability
    audio_t = torch.tensor(vec[:, :768], dtype=torch.float32)
    text_t = torch.tensor(vec[:, 768:], dtype=torch.float32)
    with torch.no_grad():
        attn_p = attn_model(audio_t, text_t).item()
    
    # 3. Ensemble (0.4/0.6 split)
    final_score = (0.4 * lgb_p) + (0.6 * attn_p)
    
    # Threshold from environment variable (default 0.35)
    threshold = float(os.getenv("HATE_THRESHOLD", 0.35))
    is_hate = bool(final_score >= threshold)
    
    return {
        "is_hateful": is_hate,
        "confidence_score": round(final_score, 4),
        "label": "Hate/Offensive" if is_hate else "Normal",
        "meta_threshold": threshold
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
