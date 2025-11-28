# Production ML Systems

> **Deploying ML models** - Serving, monitoring, MLOps basics

---

## 🎯 Model Serving

### Flask API
```python
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X = pd.DataFrame(data)
    predictions = model.predict(X)
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

---

## 📊 Monitoring

```python
# Log predictions
import logging

logging.basicConfig(filename='predictions.log', level=logging.INFO)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    predictions = model.predict(pd.DataFrame(data))
    
    logging.info(f"Input: {data}, Prediction: {predictions}")
    
    return jsonify({'predictions': predictions.tolist()})
```

---

## 📈 A/B Testing

```python
# Simple A/B test
import random

@app.route('/predict', methods=['POST'])
def predict():
    # Randomly assign to model A or B
    if random.random() < 0.5:
        model = model_a
        version = 'A'
    else:
        model = model_b
        version = 'B'
    
    predictions = model.predict(pd.DataFrame(request.json))
    
    logging.info(f"Version: {version}, Prediction: {predictions}")
    
    return jsonify({'predictions': predictions.tolist()})
```

---

**Deploy and monitor ML models in production!**
