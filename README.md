# 🏥 MedPredict — Clinical AI Decision Support System

A Flask + React web application for liver disease and diabetes risk prediction,
with uncertainty estimation (Bayesian Neural Network), doctor review flags,
explainability, EDA dashboards, and PDF export.

---

## 🚀 Run locally (any computer)

```bash
# 1. Clone your repo
git clone https://github.com/YOUR-USERNAME/medpredict.git
cd medpredict

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your datasets into the data/ folder
#    data/Indian_Liver_Patient_Dataset__ILPD_.csv
#    data/diabetes.csv

# 4. Run
python medpredict.py

# 5. Open browser
#    http://localhost:5000
```

**Login:** `doctor / medpredict123` or `admin / admin123`

---

## 🌐 Run on any device (free, 24/7)

Use **Render.com** (free tier):

1. Push this repo to GitHub
2. Go to [render.com](https://render.com) → New Web Service
3. Connect your GitHub repo
4. Set **Start command:** `python medpredict.py`
5. Set **Build command:** `pip install -r requirements.txt`
6. Click Deploy → get a public URL

Or use **Railway.app** (also free):
1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Set `python medpredict.py` as start command

---

## 📁 Project Structure

```
medpredict/
├── medpredict.py          ← All Python logic (Flask + ML + BNN)
├── test_medpredict.py     ← 51 unit tests (pytest)
├── requirements.txt       ← Python dependencies
├── CODE_QUALITY_REPORT.md ← Tool metrics (Ruff, Radon, Pytest)
├── README.md              ← This file
├── data/
│   ├── Indian_Liver_Patient_Dataset__ILPD_.csv
│   └── diabetes.csv
└── static/
    └── index.html         ← Frontend (served by Flask)
```

---

## 🧪 Run code quality tools

```bash
# Linting (style + unused imports)
ruff check medpredict.py

# Complexity check (aim for grade A or B)
radon cc medpredict.py -s -a

# Maintainability index (aim for grade A)
radon mi medpredict.py -s

# Run all tests
pytest test_medpredict.py -v

# Test coverage report
pytest test_medpredict.py --cov=medpredict --cov-report=term-missing

# Mutation testing (pick one function at a time)
mutmut run --paths-to-mutate medpredict.py \
  --runner "pytest test_medpredict.py::TestCapOutliers -x -q"
mutmut results
```

---

## 🔬 Models used

| Model | Type | Dataset |
|-------|------|---------|
| Random Forest | Bagging (ensemble) | Both |
| Gradient Boosting | Boosting | Both |
| Bagging Classifier | Bagging | Both |
| **Bayesian Neural Network** | MC-Dropout (uncertainty) | Both |

Best model is selected automatically by cross-validated AUC score.

---

## 📊 Features

- ✅ Login page (doctor / admin / nurse roles)
- ✅ Home dashboard (dataset stats, pipeline diagram)
- ✅ Prediction form for Liver Disease & Diabetes
- ✅ Risk gauge + MC uncertainty bar
- ✅ Doctor review flag (triggers when σ > 0.10)
- ✅ Feature importance / explainability chart
- ✅ Monte Carlo scatter (50 stochastic passes)
- ✅ Model comparison chart
- ✅ EDA section (age dist, scatter, histograms, correlation)
- ✅ Feature reference table
- ✅ PDF report download
- ✅ Fields reset after each prediction
- ✅ Fully responsive (phone, tablet, laptop)
- ✅ Hospital colour theme (white, teal, red)
