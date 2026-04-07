import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline

class TCAEngine:
    def __init__(self):
        self.pipe = None

    def load_model(self):
        if self.pipe is None:
            model_id = "MathRaaj/T1_bert_nli_2"
            self.pipe = pipeline("text-classification", model=model_id, device=-1)
        return self.pipe

    def process(self, text: str):
        if not text:
            return {"tca_label": "Safe Social Context", "tca_confidence": 1.0}
        
        pipe = self.load_model()
        res = pipe(text)[0]
        label = res['label'].lower()
        score = res['score']
        
        # Mapping logic from monolith
        if 'label_1' in label or 'hate' in label:
            tca_label = "Hostile Context Detected"
        else:
            tca_label = "Safe Social Context"
            
        return {
            "tca_label": tca_label,
            "tca_confidence": float(score)
        }

app = FastAPI(title="AudioGuard TCA-Svc")
engine = TCAEngine()

class TextRequest(BaseModel):
    text: str

@app.post("/analyze")
def analyze_text(req: TextRequest):
    try:
        return engine.process(req.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
