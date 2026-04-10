import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, BertForSequenceClassification

class TCAEngine:
    def __init__(self):
        print("TCA-Svc: Loading BERT Model during startup...")
        model_id = "MathRaaj/T1_bert_nli_3"
        
        # Load model and tokenizer directly to access hidden states
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = BertForSequenceClassification.from_pretrained(model_id)
        self.model.eval()
        print("TCA-Svc: Model Loaded Successfully!")

    def process(self, text: str):
        if not text:
            # Return neutral embedding (zeros) if no text
            return {
                "tca_label": "Safe Social Context", 
                "tca_confidence": 1.0,
                "embedding": [0.0] * 768
            }
        
        # Tokenize and run inference
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            
            # Embedding: Extract the [CLS] token (index 0) from the last hidden state
            # This is the standard 768-dim representation used for classification
            last_hidden_state = outputs.hidden_states[-1]
            embedding = last_hidden_state[:, 0, :].squeeze().tolist()
            
        score, idx = torch.max(probs, dim=-1)
        label = self.model.config.id2label[idx.item()].lower()
        
        # Mapping logic from monolith
        if 'label_1' in label or 'hate' in label:
            tca_label = "Hostile Context Detected"
        else:
            tca_label = "Safe Social Context"
            
        return {
            "tca_label": tca_label,
            "tca_confidence": float(score),
            "embedding": embedding
        }

app = FastAPI(title="AudioGuard TCA-Svc")
engine = None

@app.on_event("startup")
async def startup_event():
    global engine
    engine = TCAEngine()

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(req: TextRequest):
    try:
        return engine.process(req.text)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
