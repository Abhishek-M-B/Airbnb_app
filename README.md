# 🏠 NYC Airbnb Price Predictor

A machine learning web app that predicts nightly Airbnb prices in New York City using XGBoost.

## ⚡ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| ML Model | XGBoost Pipeline |
| Frontend | HTML, CSS, JavaScript |
| Data | 30,475 NYC Airbnb listings |

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/Abhishek-M-B/Airbnb_app.git
cd Airbnb_app

# 2. Create virtual environment
python -m venv airbnb_venv
airbnb_venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add model file
# Place airbnb_xgboost_model.pkl in the root folder

# 5. Run
python app.py
```

Open `http://localhost:5000` in your browser.

---

## 🧠 Model Performance

| Metric | Score |
|--------|-------|
| Algorithm | XGBoost |
| R² Score | 0.60 |
| MAE | 0.29 |
| CV Folds | 5-fold |

---

## 📁 Project Structure

```
Airbnb_app/
├── app.py                 # Flask backend
├── requirements.txt       # Dependencies
├── templates/
│   └── index.html         # Frontend UI
└── .gitignore
```
---

## 📸 Demo

> ![Demo](static/airbnb_demo.gif)

---
---

> ⚠️ **Note:** `airbnb_xgboost_model.pkl` is not included in the repo due to file size. Train the model using the notebook and place it in the root folder before running.
