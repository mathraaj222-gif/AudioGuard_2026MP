import os
import tempfile
import numpy as np

from moviepy import VideoFileClip
import whisper
import librosa
import noisereduce as nr
from transformers import pipeline, AutoModelForAudioClassification
import torch

from late_fusion_pipeline import fallback_fusion

class InferencePipeline:
    def __init__(self):
        """
        Initialize the pipeline models.
        (Mocks loading for demonstration purposes; replace with actual model loading).
        """
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        
        print("Loading Best SER Model (S4: wav2vec2-large)...")
        ser_model_path = os.path.join("outputs", "S4_wav2vec2_large")
        try:
            if os.path.exists(ser_model_path):
                self.ser_model = pipeline("audio-classification", model=ser_model_path)
                print("SER Model Loaded.")
            else:
                self.ser_model = None
                print(f"Warning: {ser_model_path} not found. Mocking SER.")
        except Exception as e:
            print(f"Warning: Failed to load SER model as HuggingFace pipeline. ({e}). Mocking SER.")
            self.ser_model = None
            
        print("Loading Best TCA Model (T3: deberta_large_nli)...")
        # Load the best TCA Model: T3_deberta_large_nli from the outputs directory
        tca_model_path = os.path.join("outputs", "T3_deberta_large_nli", "best_model")
        try:
            if os.path.exists(tca_model_path):
                self.tca_model = pipeline("text-classification", model=tca_model_path, top_k=None)
                print("TCA Model Loaded.")
            else:
                self.tca_model = None
                print(f"Warning: {tca_model_path} not found. Mocking TCA.")
        except Exception as e:
            print(f"Warning: Failed to load TCA model as HuggingFace pipeline. ({e}). Mocking TCA.")
            self.tca_model = None
        
        print("Pipeline initialized successfully.")

    def extract_audio(self, video_path, output_audio_path):
        """Extracts audio from a video file."""
        print(f"Extracting audio from {video_path}...")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_audio_path, verbose=False, logger=None)
        return output_audio_path

    def clean_audio(self, audio_path):
        """Cleans and reduces noise in audio."""
        print(f"Cleaning audio {audio_path}...")
        try:
            data, rate = librosa.load(audio_path, sr=16000)
            reduced_noise = nr.reduce_noise(y=data, sr=rate)
            return reduced_noise
        except Exception as e:
            print(f"Error cleaning audio: {e}. Returning original.")
            data, _ = librosa.load(audio_path, sr=16000)
            return data

    def transcribe_audio(self, audio_data):
        """Transcribes and translates audio to English text."""
        print(f"Transcribing audio sequence...")
        result = self.whisper_model.transcribe(audio_data, task="translate")
        return result["text"]

    def get_ser_probabilities(self, audio_data):
        """Predicts emotion probabilities."""
        print("Running SER model...")
        if self.ser_model is None:
            return np.array([0.1, 0.85, 0.0, 0.05, 0.0, 0.0, 0.0]) # Mock output
            
        preds = self.ser_model({"array": audio_data, "sampling_rate": 16000})
        # Assuming predictions return [{'label': 'ClassX', 'score': 0.9}, ...]
        # Sort or process as needed if label order matters. For now we will just use the max
        # or construct an array. Here we construct a dummy array, and put the max predicted score at idx 1 (angry) if it matches
        # but realistically you need a label to index mapping. We will return 7-class dummy-filled array for now, placing highest label at best match.
        probs = np.zeros(7)
        pred_label = preds[0]['label']
        pred_score = preds[0]['score']
        
        angry_labels = ['angry', 'anger', 'hate']
        if any(a in pred_label.lower() for a in angry_labels):
            probs[1] = pred_score # Let's say idx 1 is Hate/Anger
        else:
            probs[0] = pred_score # Let's say idx 0 is Neutral
            
        return probs

    def get_tca_probabilities(self, text):
        """Predicts hate speech probabilities from text."""
        print(f"Running TCA model on: '{text}'")
        if self.tca_model is None:
            return np.array([0.05, 0.90, 0.05, 0.0, 0.0, 0.0, 0.0]) # Mock output
            
        preds = self.tca_model(text)[0] 
        if len(preds) > 0 and isinstance(preds[0], list):
            preds = preds[0]
            
        # Returns list of dicts: [{'label': 'hate', 'score': 0.8}, {'label': 'neutral', 'score': 0.2}]
        # If it returns LABEL_X instead of text:
        probs = np.zeros(7)
        for p in preds:
            label_lower = str(p['label']).lower()
            score = float(p['score'])
            if np.isnan(score):
                score = 0.0
                
            if 'hate' in label_lower or 'offensive' in label_lower or 'label_1' in label_lower or 'label_2' in label_lower:
                probs[1] += score
            else:
                probs[0] += score
                
        # Normalize
        s = np.sum(probs)
        if s > 0: probs = probs / s
        return probs

    def process_file(self, file_path):
        """
        End-to-End Pipeline
        """
        is_video = file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        audio_path = file_path
        
        # 1. Video to Audio
        if is_video:
            temp_audio = tempfile.mktemp(suffix=".wav")
            audio_path = self.extract_audio(file_path, temp_audio)
            
        # 2. Audio Cleaning (Returns NumPy Array)
        clean_audio_data = self.clean_audio(audio_path)
        
        # 3. Transcription & Translation
        transcription = self.transcribe_audio(clean_audio_data)
        
        # 4. Model Predictions
        ser_probs = self.get_ser_probabilities(clean_audio_data)
        tca_probs = self.get_tca_probabilities(transcription)
        
        # 5. Fusion
        final_class, merged_probs = fallback_fusion(tca_probs, ser_probs, tca_threshold=0.7)
        
        is_hatespeech = bool(final_class == 1) # Assuming class 1 is Hate Speech
        confidence = float(np.max(merged_probs))
        
        return {
            "transcription": transcription,
            "detected_emotion": "Anger/Aggressive" if np.argmax(ser_probs) == 1 else "Neutral",
            "is_hatespeech": is_hatespeech,
            "confidence": f"{confidence * 100:.2f}%",
            "tca_confidence": f"{np.max(tca_probs)*100:.2f}%",
            "ser_confidence": f"{np.max(ser_probs)*100:.2f}%"
        }

# For testing locally
if __name__ == "__main__":
    pipeline = InferencePipeline()
    result = pipeline.process_file("dummy_video.mp4")
    print("\n--- Pipeline Result ---")
    print(result)
