# ============================================================
#   CreditLens AI - Loan Default Risk Scoring System
#   Flask Backend — connects trained RF model to frontend
# ============================================================

from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# ── Load saved model & columns ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model   = joblib.load(os.path.join(BASE_DIR, "loan_model.pkl"))
columns = joblib.load(os.path.join(BASE_DIR, "model_columns.pkl"))

print("✅ Model loaded successfully!")
print(f"   Features expected: {list(columns)}")


# ── Helper: build feature row from form input ────────────────
def build_features(data: dict) -> pd.DataFrame:
    age           = float(data["age"])
    income        = float(data["income"])
    loan_amount   = float(data["loan_amount"])
    loan_term     = float(data["loan_term"])
    interest_rate = float(data["interest_rate"])
    months_emp    = float(data["months_emp"])
    credit_score  = float(data["credit_score"])
    credit_lines  = float(data["credit_lines"])
    dti           = float(data["dti"])
    has_mortgage  = int(data["mortgage"])
    has_cosigner  = int(data["cosigner"])
    emp_type      = data["employment"]

    row = {col: 0 for col in columns}

    row["Age"]            = age
    row["Income"]         = income
    row["LoanAmount"]     = loan_amount
    row["LoanTerm"]       = loan_term
    row["InterestRate"]   = interest_rate
    row["MonthsEmployed"] = months_emp
    row["CreditScore"]    = credit_score
    row["NumCreditLines"] = credit_lines
    row["DTIRatio"]       = dti

    if "HasMortgage_Yes" in row:
        row["HasMortgage_Yes"] = has_mortgage
    if "HasCoSigner_Yes" in row:
        row["HasCoSigner_Yes"] = has_cosigner

    emp_map = {
        "part":       "EmploymentType_Part-time",
        "self":       "EmploymentType_Self-employed",
        "unemployed": "EmploymentType_Unemployed",
    }
    col_name = emp_map.get(emp_type)
    if col_name and col_name in row:
        row[col_name] = 1

    return pd.DataFrame([row])


# ── Route: serve the HTML frontend ───────────────────────────
@app.route("/")
def index():
    html_path = os.path.join(BASE_DIR, "loan_risk_frontend.html")
    with open(html_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Safely inject API override script just before </body>
    api_js = """
<script>
// Override the predict function to call the real Flask /predict API
window.predict = function() {
  const req = ['age','income','employment','months_emp','loan_amount',
               'loan_term','credit_score','credit_lines','dti','mortgage','cosigner'];
  let ok = true;
  req.forEach(id => {
    if (document.getElementById(id).value === '') ok = false;
  });
  if (!ok) { showToast('Please fill in all required fields.', false); return; }

  document.getElementById('btn-text').style.display = 'none';
  document.getElementById('loader').style.display   = 'block';

  const payload = {
    age:           document.getElementById('age').value,
    income:        document.getElementById('income').value,
    employment:    document.getElementById('employment').value,
    months_emp:    document.getElementById('months_emp').value,
    loan_amount:   document.getElementById('loan_amount').value,
    loan_term:     document.getElementById('loan_term').value,
    interest_rate: document.getElementById('interest_rate').value,
    credit_score:  document.getElementById('credit_score').value,
    credit_lines:  document.getElementById('credit_lines').value,
    dti:           document.getElementById('dti').value,
    mortgage:      document.getElementById('mortgage').value,
    cosigner:      document.getElementById('cosigner').value
  };

  fetch('/predict', {
    method:  'POST',
    headers: {'Content-Type': 'application/json'},
    body:    JSON.stringify(payload)
  })
  .then(r => r.json())
  .then(result => {
    document.getElementById('btn-text').style.display = 'block';
    document.getElementById('loader').style.display   = 'none';

    if (result.error) { showToast(result.error, false); return; }

    const v = {
      age:    +payload.age,
      income: +payload.income,
      emp:     payload.employment,
      months: +payload.months_emp,
      loan:   +payload.loan_amount,
      term:   +payload.loan_term,
      rate:   +payload.interest_rate,
      cs:     +payload.credit_score,
      lines:  +payload.credit_lines,
      dti:    +payload.dti,
      mort:   +payload.mortgage,
      co:     +payload.cosigner
    };

    displayResult(result.risk_percent, v);
    showToast('Model prediction complete. Probability: ' + result.probability.toFixed(3));
  })
  .catch(err => {
    document.getElementById('btn-text').style.display = 'block';
    document.getElementById('loader').style.display   = 'none';
    showToast('Server error: ' + err.message, false);
  });
};
</script>
</body>"""

    html = html.replace("</body>", api_js)
    return html


# ── Route: POST /predict ──────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        required = ["age","income","employment","months_emp","loan_amount",
                    "loan_term","interest_rate","credit_score","credit_lines",
                    "dti","mortgage","cosigner"]
        for field in required:
            if field not in data or data[field] == "" or data[field] is None:
                return jsonify({"error": f"Missing field: {field}"}), 400

        X = build_features(data)

        probability  = float(model.predict_proba(X)[0][1])
        prediction   = int(model.predict(X)[0])
        risk_percent = round(probability * 100, 1)

        if risk_percent >= 55:
            risk_level = "High Risk"
        elif risk_percent >= 30:
            risk_level = "Medium Risk"
        else:
            risk_level = "Low Risk"

        return jsonify({
            "probability":  probability,
            "risk_percent": risk_percent,
            "prediction":   prediction,
            "risk_level":   risk_level,
            "decision":     "Reject" if prediction == 1 else "Approve"
        })

    except KeyError as e:
        return jsonify({"error": f"Feature mismatch: {str(e)} — check model_columns.pkl"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Route: GET /model-info ────────────────────────────────────
@app.route("/model-info")
def model_info():
    return jsonify({
        "model_type":    type(model).__name__,
        "n_estimators":  model.n_estimators,
        "max_depth":     model.max_depth,
        "n_features":    model.n_features_in_,
        "feature_names": list(columns),
        "classes":       list(model.classes_.astype(str)),
    })


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🚀 CreditLens server starting...")
    print("   Open in browser: http://127.0.0.1:5000\n")
    app.run(debug=True, port=5000)
