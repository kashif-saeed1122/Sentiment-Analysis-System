# CHEAT SHEET - Just the Commands You Need

## 💡 Two Ways to Run This

### Option A: All Local (Needs 8GB+ RAM)
```bash
python train_pipeline.py
```
## 🚀 First Time Setup (Do This Once)

```bash
cd sentiment-analysis-system
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## ⚡ Quick Start (Everything at Once)

```bash
python train_pipeline.py
```

**What happens:**
- Downloads data (2 min)
- Trains ML model (1 min)  
- Trains DL model (15 min)
- Creates comparison charts

**Done!** Now you have trained models.

---

## 🌐 Start the API

### Terminal 1: Start Server
```bash
cd src/api
python main.py
```

Keep this running!

### Terminal 2: Test It
```bash
# Easy way - automated tests
python test_api.py

# Or manually test
curl -X POST http://localhost:8000/predict-ml \
  -H "Content-Type: application/json" \
  -d '{"text": "This is great!"}'
```

---

## 📝 Common Commands

| What | Command |
|------|---------|
| Train everything | `python train_pipeline.py` |
| Start API | `cd src/api && python main.py` |
| Test API | `python test_api.py` |
| Check health | `curl http://localhost:8000/healthcheck` |
| View analysis | `jupyter notebook notebooks/model_comparison.ipynb` |

---

## 🎯 Step-by-Step (If You Want Control)

```bash
# 1. Get data
python src/data/collect_data.py

# 2. Clean data
python src/preprocessing/text_preprocessing.py

# 3. Train ML model (fast)
python src/models/train_ml.py

# 4. Train DL model (slow - optional!)
python src/models/train_dl.py

# 5. Compare models
python src/evaluation/evaluate_models.py
```

---

## 🔧 Example API Usage in Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict-ml",
    json={"text": "This product is amazing!"}
)

result = response.json()
print(f"Sentiment: {result['sentiment']}")
print(f"Confidence: {result['confidence']}")
```

---

## 📂 Where Are My Files?

```
data/raw/raw_sentiment_data.csv          ← Downloaded data
data/processed/train.csv                 ← Training data
saved_models/ml/logistic_model.pkl       ← ML model
saved_models/dl/best_model/              ← DL model
logs/predictions.db                      ← API logs
```

---

## ⚠️ Troubleshooting

**"Module not found"**
→ `pip install -r requirements.txt`

**"No such file or directory"**
→ Make sure you're in `sentiment-analysis-system` folder

**"Models not loaded"**
→ Run `python train_pipeline.py` first

**"DL training too slow"**
→ It's normal (10-15 min). You can skip it!

---

## 💡 Tips

- Always run from project root directory
- You can skip DL training - ML model is enough
- API runs on port 8000
- Predictions are logged to SQLite database
- Check logs/ folder if something goes wrong

---

**That's it! Start with: `python train_pipeline.py` 🎉**