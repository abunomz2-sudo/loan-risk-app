# 🏦 RiskIQ — AI Loan Default Risk Scoring System

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=flat-square&logo=scikit-learn)
![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7?style=flat-square&logo=render)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> An AI-powered web application that predicts loan default risk using a trained Random Forest classifier. Enter an applicant's financial details and instantly receive a calibrated default probability score with key risk driver analysis.

---

## 🌐 Live Demo

🔗 **[https://loan-risk-app.onrender.com](https://loan-risk-app.onrender.com)**

> ⚠️ Free tier may take 30–60 seconds to wake up on first visit.

---

## 📸 Screenshots

### Risk Assessment
![Risk Assessment Tab](https://via.placeholder.com/900x500?text=Risk+Assessment+Screenshot)

### Training Metrics
![Training Metrics Tab](https://via.placeholder.com/900x500?text=Training+Metrics+Screenshot)

> 💡 Replace the placeholder images above with real screenshots of your app.

---

## ✨ Features

- 🤖 **Real ML Predictions** — Random Forest model trained on 12,000 loan records
- 📊 **Risk Gauge & Score** — Animated dial showing Low / Medium / High risk
- 🔍 **Key Risk Drivers** — Visual breakdown of the 6 most influential factors
- 📈 **Training Metrics Tab** — ROC curve, confusion matrix, feature importance charts
- 🎨 **Modern UI** — Glass-morphism design with smooth animations
- 📱 **Responsive** — Works on desktop and mobile

---

## 🧠 Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Training Records | 12,000 |
| Test Accuracy | **76.1%** |
| ROC-AUC Score | **0.80** |
| Number of Trees | 200 |
| Max Depth | 8 |
| Class Balancing | Cost-sensitive (balanced_subsample) |
| Train/Test Split | 80% / 20% Stratified |

### Top Predictive Features
1. **Credit Score** — 34% importance
2. **Interest Rate** — 32% importance
3. **Annual Income** — 6% importance
4. **Loan Amount** — 5% importance
5. **Has Co-Signer** — 5% importance
6. **DTI Ratio** — 4% importance

---

## 🗂️ Project Structure

```
loan-risk-app/
│
├── Frontend.py                 # Flask backend — routes & ML prediction logic
├── loan_risk_frontend.html     # Frontend UI — all HTML, CSS, JS in one file
├── loan_model.pkl              # Trained Random Forest model
├── model_columns.pkl           # Feature column names from training
├── requirements.txt            # Python dependencies
├── Procfile                    # Gunicorn start command for Render
└── README.md                   # You are here!
```

---

## 🚀 Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/loan-risk-app.git
cd loan-risk-app
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Flask server
```bash
python Frontend.py
```

### 4. Open in browser
```
http://127.0.0.1:5000
```

---

## 📡 API Reference

### `POST /predict`

Accepts applicant data and returns a default risk prediction.

**Request Body (JSON):**
```json
{
  "age": 35,
  "income": 600000,
  "employment": "full",
  "months_emp": 36,
  "loan_amount": 500000,
  "loan_term": 36,
  "interest_rate": 8.5,
  "credit_score": 680,
  "credit_lines": 4,
  "dti": 0.35,
  "mortgage": 0,
  "cosigner": 0
}
```

**Employment values:** `full` · `part` · `self` · `unemployed`

**Response (JSON):**
```json
{
  "probability": 0.312,
  "risk_percent": 31.2,
  "prediction": 0,
  "risk_level": "Medium Risk",
  "decision": "Approve"
}
```

### `GET /model-info`

Returns metadata about the loaded model.

```json
{
  "model_type": "RandomForestClassifier",
  "n_estimators": 300,
  "max_depth": 10,
  "n_features": 14,
  "feature_names": ["Age", "Income", "LoanAmount", "..."],
  "classes": ["0", "1"]
}
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| ML Model | scikit-learn (RandomForestClassifier) |
| Data Processing | pandas, NumPy |
| Model Persistence | joblib |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Charts | Chart.js 4.4 |
| Fonts | Google Fonts (Playfair Display, Plus Jakarta Sans) |
| Deployment | Render (free tier) |
| Version Control | GitHub |

---

## ⚙️ Input Fields Explained

| Field | Description | Range |
|---|---|---|
| Age | Applicant age in years | 18 – 90 |
| Annual Income | Yearly income in ₹ | Any positive value |
| Employment Type | Job category | Full-time / Part-time / Self-employed / Unemployed |
| Months Employed | Duration at current job | 0+ |
| Loan Amount | Requested loan in ₹ | Any positive value |
| Loan Term | Repayment period in months | 6 – 360 |
| Interest Rate | Annual interest rate | 1% – 36% |
| Credit Score | CIBIL / credit bureau score | 300 – 850 |
| Num. Credit Lines | Open credit accounts | 0+ |
| DTI Ratio | Debt-to-Income ratio | 0.00 – 1.00 |
| Has Mortgage | Existing mortgage | Yes / No |
| Has Co-Signer | Loan co-signer present | Yes / No |

---

## 📊 Model Performance

```
Classification Report:
                  Precision   Recall   F1-Score   Support
  0 (No Default)    0.90       0.79      0.84      1,933
  1 (Default)       0.43       0.64      0.51        467
  Macro Avg         0.66       0.72      0.68      2,400
  Weighted Avg      0.81       0.76      0.78      2,400

Confusion Matrix:
  True Negative:  1,526  |  False Positive:  407
  False Negative:   166  |  True Positive:   301
```

---

## 🔮 Future Improvements

- [ ] Add user authentication for loan officers
- [ ] Export predictions as PDF report
- [ ] Batch prediction via CSV upload
- [ ] Integrate XGBoost / LightGBM for better AUC
- [ ] Add SHAP values for explainability
- [ ] Connect to a real database for prediction history

---

## 👨‍💻 Author

**Abuthahir**
- GitHub: [@ABUTHAHIR](https://github.com/ABUTHAHIR)

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use, modify, and distribute.

---

## 🙏 Acknowledgements

- [scikit-learn](https://scikit-learn.org/) for the Random Forest implementation
- [Chart.js](https://www.chartjs.org/) for beautiful charts
- [Render](https://render.com/) for free hosting
- [Google Fonts](https://fonts.google.com/) for typography

---

<p align="center">Made with ❤️ for AI-powered credit intelligence</p>
