from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import torch
import time
import sqlite3
from datetime import datetime
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import uvicorn

app = FastAPI(
    title="Sentiment Analysis API",
    description="Production-ready sentiment analysis with ML and DL models",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    label: int
    model_type: str
    inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    ml_model_loaded: bool
    dl_model_loaded: bool
    timestamp: str

class ModelManager:
    def __init__(self):
        self.ml_model = None
        self.ml_vectorizer = None
        self.dl_model = None
        self.dl_tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        self._init_db()
        
    def _init_db(self):
        """Initialize SQLite database for logging"""
        db_path = Path('logs/predictions.db')
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                input_text TEXT NOT NULL,
                prediction TEXT NOT NULL,
                label INTEGER NOT NULL,
                confidence REAL NOT NULL,
                model_used TEXT NOT NULL,
                inference_time_ms REAL NOT NULL
            )
        ''')
        conn.commit()
        conn.close()
    
    def log_prediction(self, text, prediction, label, confidence, model_type, inference_time):
        """Log prediction to database"""
        conn = sqlite3.connect('logs/predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (timestamp, input_text, prediction, label, confidence, model_used, inference_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            text[:500],
            prediction,
            label,
            confidence,
            model_type,
            inference_time
        ))
        conn.commit()
        conn.close()
    
    def load_ml_model(self):
        """Load ML model and vectorizer"""
        try:
            with open('saved_models/ml/tfidf_vectorizer.pkl', 'rb') as f:
                self.ml_vectorizer = pickle.load(f)
            
            with open('saved_models/ml/logistic_model.pkl', 'rb') as f:
                self.ml_model = pickle.load(f)
            
            return True
        except Exception as e:
            print(f"Error loading ML model: {e}")
            return False
    
    def load_dl_model(self):
        """Load DL model and tokenizer"""
        try:
            model_path = 'saved_models/dl/best_model'
            self.dl_model = DistilBertForSequenceClassification.from_pretrained(model_path)
            self.dl_tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            self.dl_model.to(self.device)
            self.dl_model.eval()
            return True
        except Exception as e:
            print(f"Error loading DL model: {e}")
            return False
    
    def predict_ml(self, text):
        """Predict sentiment using ML model"""
        if not self.ml_model or not self.ml_vectorizer:
            raise HTTPException(status_code=500, detail="ML model not loaded")
        
        start_time = time.time()
        
        X = self.ml_vectorizer.transform([text])
        prediction = self.ml_model.predict(X)[0]
        proba = self.ml_model.predict_proba(X)[0]
        confidence = float(max(proba))
        
        inference_time = (time.time() - start_time) * 1000
        
        sentiment = self.label_map[prediction]
        
        self.log_prediction(text, sentiment, int(prediction), confidence, 'ml', inference_time)
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'label': int(prediction),
            'model_type': 'ml',
            'inference_time_ms': inference_time
        }
    
    def predict_dl(self, text):
        """Predict sentiment using DL model"""
        if not self.dl_model or not self.dl_tokenizer:
            raise HTTPException(status_code=500, detail="DL model not loaded")
        
        start_time = time.time()
        
        encoding = self.dl_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.dl_model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = float(probs[0][prediction])
        
        inference_time = (time.time() - start_time) * 1000
        
        sentiment = self.label_map[prediction]
        
        self.log_prediction(text, sentiment, prediction, confidence, 'dl', inference_time)
        
        return {
            'text': text,
            'sentiment': sentiment,
            'confidence': confidence,
            'label': prediction,
            'model_type': 'dl',
            'inference_time_ms': inference_time
        }

model_manager = ModelManager()

@app.on_event("startup")
async def startup_event():
    """Load models on startup"""
    print("Loading models...")
    ml_loaded = model_manager.load_ml_model()
    dl_loaded = model_manager.load_dl_model()
    print(f"ML model loaded: {ml_loaded}")
    print(f"DL model loaded: {dl_loaded}")

@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "message": "Sentiment Analysis API",
        "endpoints": {
            "ml": "/predict-ml",
            "dl": "/predict-dl",
            "health": "/healthcheck"
        }
    }

@app.post("/predict-ml", response_model=PredictionResponse)
async def predict_ml(request: PredictionRequest):
    """Predict sentiment using ML model"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = model_manager.predict_ml(request.text)
    return PredictionResponse(**result)

@app.post("/predict-dl", response_model=PredictionResponse)
async def predict_dl(request: PredictionRequest):
    """Predict sentiment using DL model"""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    result = model_manager.predict_dl(request.text)
    return PredictionResponse(**result)

@app.get("/healthcheck", response_model=HealthResponse)
async def healthcheck():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        ml_model_loaded=model_manager.ml_model is not None,
        dl_model_loaded=model_manager.dl_model is not None,
        timestamp=datetime.now().isoformat()
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)