# Neonatal Sepsis Prediction API

Production-ready FastAPI backend for XGBoost neonatal sepsis prediction.

---

## 1. FOLDER STRUCTURE

```
neonatal_sepsis_api/
│
├── main.py                          # FastAPI app — all routes + preprocessing pipeline
├── requirements.txt                 # Pinned dependencies (MUST match training env)
├── Procfile                         # For Render / Heroku deployment
├── render.yaml                      # Render Infrastructure-as-Code config
├── runtime.txt                      # Python version pin
├── test_api.py                      # Full integration test suite
├── .gitignore
├── README.md
│
└── models/                          # ← PUT YOUR .pkl FILES HERE
    ├── xgb_sepsis_model_final.pkl   # ← Trained XGBoost model
    └── sepsis_preprocessors_final.pkl  # ← All preprocessors
```

---

## 2. PLACE YOUR MODEL FILES

**This is the most common deployment mistake.**

Copy your `.pkl` files into the `models/` folder:

```bash
# From your project directory where you trained the model:
cp xgb_sepsis_model_final.pkl     neonatal_sepsis_api/models/
cp sepsis_preprocessors_final.pkl neonatal_sepsis_api/models/
```

The `models/` folder MUST contain exactly:
- `xgb_sepsis_model_final.pkl`
- `sepsis_preprocessors_final.pkl`

---

## 3. LOCAL SETUP

### Step 1: Create and activate virtual environment

```bash
cd neonatal_sepsis_api
python3 -m venv venv
source venv/bin/activate          # Linux / Mac
# venv\Scripts\activate           # Windows
```

### Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

**Why pinned versions?** `scikit-learn` version MUST match the version used
when you saved the preprocessors. The pkl was saved with scikit-learn 1.6.1.
If versions mismatch, imputers/scalers can produce silent wrong results.

### Step 3: Start the server (development)

```bash
uvicorn main:app --reload --port 8000
```

You should see:
```
INFO: === STARTUP: Loading model and preprocessors ===
INFO: ✅ Model loaded     : XGBClassifier
INFO: ✅ Total features   : 19
INFO: ✅ Feature list     : ['ga', 'bw', 'enr_leth', ...]
INFO: Uvicorn running on http://0.0.0.0:8000
```

If you see `FileNotFoundError`, your pkl files are not in `models/`.

---

## 4. TESTING LOCALLY

### Option A: Run the test suite

```bash
# Terminal 1 — start server
uvicorn main:app --reload

# Terminal 2 — run all tests
python test_api.py
```

Expected output:
```
✅ PASS — Root endpoint
✅ PASS — Health check
✅ PASS — Predict — full 19 features
✅ PASS — Predict — partial (CRP + TLC)
✅ PASS — Predict — empty input (all NaN)
...
RESULTS: 10/10 passed
🎉 All tests passed! Safe to deploy.
```

### Option B: Swagger UI

Open in browser: **http://localhost:8000/docs**

1. Click `POST /predict` → `Try it out`
2. Paste one of the example requests below
3. Click `Execute`

---

## 5. JSON REQUEST EXAMPLES

### Example 1: Full input (all 19 features)
```json
{
    "ga": 32.0,
    "bw": 1500.0,
    "enr_leth": 1,
    "enr_cry": 0,
    "enr_refl": 1,
    "enr_fever": 1,
    "enr_tachyc": 1,
    "enr_tachyp": 1,
    "enr_apn": 0,
    "enr_retr": 1,
    "enr_cyan": 0,
    "enr_abd": 1,
    "enr_puls": 0,
    "enr_hi_cry": 0,
    "enr_cxr": 1,
    "enr_fio2": 40.0,
    "enr_crp_val": 25.5,
    "enr_tlc_val": 22.0,
    "age_onset": 48.0
}
```

### Example 2: Minimal — only labs (17 features imputed)
```json
{
    "enr_crp_val": 12.5,
    "enr_tlc_val": 18.0
}
```

### Example 3: Premature infant with signs
```json
{
    "ga": 28.0,
    "bw": 900.0,
    "enr_fever": 1,
    "enr_leth": 1,
    "enr_crp_val": 30.0
}
```

### Example 4: Empty (all imputed from training medians)
```json
{}
```

---

## 6. EXPECTED RESPONSE FORMAT

```json
{
    "sepsis_probability": 0.7823,
    "sepsis_probability_pct": "78.2%",
    "risk_category": "High",
    "risk_message": "Immediate clinical evaluation required. Consider empirical antibiotics.",
    "prediction": 1,
    "features_received": 5,
    "features_missing": 14,
    "features_used": ["ga", "bw", "enr_leth", "enr_cry", ...],
    "warning": null
}
```

---

## 7. RENDER DEPLOYMENT

### Prerequisites
- GitHub account with your project pushed to a repo
- Render account (free at render.com)

### Step-by-step

**Step 1:** Push your project to GitHub
```bash
cd neonatal_sepsis_api
git init
git add .
git commit -m "Initial commit — sepsis prediction API"
git remote add origin https://github.com/YOUR_USERNAME/neonatal-sepsis-api.git
git push -u origin main
```

**IMPORTANT:** The `models/` folder with your `.pkl` files MUST be committed.
Files > 100MB need Git LFS (`git lfs track "*.pkl"`).

**Step 2:** Create Render service
1. Go to https://render.com → New → Web Service
2. Connect your GitHub repo
3. Render auto-detects `render.yaml` and pre-fills settings
4. Click **Create Web Service**

**Step 3:** Render auto-detects and runs:
```
Build: pip install -r requirements.txt
Start: gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120 --bind 0.0.0.0:$PORT
```

**Step 4:** Your API is live at:
```
https://neonatal-sepsis-api.onrender.com
```

**Step 5:** Test deployment health:
```bash
curl https://neonatal-sepsis-api.onrender.com/health
```

### Environment Variables on Render
In Render Dashboard → Your Service → Environment:

| Key | Value |
|-----|-------|
| LOG_LEVEL | INFO |
| MODEL_DIR | models |
| MAX_BATCH | 100 |

---

## 8. PRODUCTION START COMMAND

```bash
# Local production (no reload):
gunicorn main:app --workers 2 --worker-class uvicorn.workers.UvicornWorker --timeout 120 --bind 0.0.0.0:8000

# Development (with auto-reload):
uvicorn main:app --reload --port 8000
```

---

## 9. COMMON ERRORS AND FIXES

### Error: `FileNotFoundError: Model file not found: models/xgb_sepsis_model_final.pkl`
**Fix:** Copy your `.pkl` files to the `models/` folder.
```bash
cp /path/to/xgb_sepsis_model_final.pkl models/
cp /path/to/sepsis_preprocessors_final.pkl models/
```

### Error: `InconsistentVersionWarning: Trying to unpickle estimator from version 1.6.1 when using version X.Y.Z`
**Fix:** Your scikit-learn version doesn't match. Install the exact version:
```bash
pip install scikit-learn==1.6.1
```

### Error: `Feature shape mismatch, expected: 19, got N`
**Fix:** The model was trained with 19 features but gets a different number.
This is now impossible with the current code because:
1. Missing features are added as NaN (Step 2 of preprocessing)
2. Columns are always reindexed to training order (Step 3)

If you still see this, check that your `.pkl` files are the ones saved AFTER
the final feature elimination step (not an earlier version).

### Error: `422 Unprocessable Entity`
**Fix:** The request body has an incorrect data type. Check that numeric fields
are passed as numbers, not strings. Example wrong: `"ga": "thirty-two"`.
Correct: `"ga": 32.0`.

### Error: `503 Service Unavailable — Model not loaded`
**Fix:** Server is starting up. Wait 5-10 seconds on cold start. If persistent,
check server logs — a startup error (bad pkl path or version mismatch) prevents
the model from loading.

### Error: `XGBoost feature_names mismatch`
**Fix:** Already handled by the `reindex(columns=features)` in Step 9 of
`run_inference_preprocessing()`. If you see this error, verify that the pkl
files are the correct final versions saved with `joblib.dump()` at the end of
training.

---

## 10. HOW FEATURE MISMATCH IS PREVENTED PERMANENTLY

The core problem and its solution:

```
PROBLEM: Model was trained with features ['ga', 'bw', ...] in a specific order.
         At inference, if any feature is missing or in wrong order → crash.

SOLUTION (in run_inference_preprocessing):

   Step 2: for col in training_features:
               if col not in request:
                   df[col] = NaN  ← adds missing columns as NaN

   Step 3: df = df.reindex(columns=training_features)  ← enforces exact order

   Step 5: df[num_cols] = num_imputer.transform(...)  ← fills NaN with medians

   Step 9: df_final = df_final.reindex(columns=features)  ← double-check order
```

This makes feature mismatch **structurally impossible**.

---

## 11. FEATURE REFERENCE

| Feature | Description | Type | Range |
|---------|-------------|------|-------|
| ga | Gestational age | float | weeks |
| bw | Birth weight | float | grams |
| enr_leth | Lethargy | binary | 0/1 |
| enr_cry | Abnormal cry | binary | 0/1 |
| enr_refl | Poor reflexes | binary | 0/1 |
| enr_fever | Fever | binary | 0/1 |
| enr_tachyc | Tachycardia | binary | 0/1 |
| enr_tachyp | Tachypnoea | binary | 0/1 |
| enr_apn | Apnoea | binary | 0/1 |
| enr_retr | Retractions | binary | 0/1 |
| enr_cyan | Cyanosis | binary | 0/1 |
| enr_abd | Abdominal distension | binary | 0/1 |
| enr_puls | Abnormal pulse | binary | 0/1 |
| enr_hi_cry | High-pitched cry | binary | 0/1 |
| enr_cxr | Abnormal CXR | binary | 0/1 |
| enr_fio2 | FiO2 requirement | float | % |
| enr_crp_val | CRP value | float | mg/dL |
| enr_tlc_val | TLC value | float | ×10⁹/L |
| age_onset | Age at symptom onset | float | hours |

All features are optional. Missing features are imputed with training medians.
