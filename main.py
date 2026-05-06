# =============================================================================
# main.py — Neonatal Sepsis Prediction API
# Production FastAPI backend for XGBoost inference
# Author: ML Deployment Engineer
# =============================================================================
# CRITICAL DESIGN DECISIONS:
#   - Preprocessors are loaded ONCE at startup (not per-request)
#   - Inference uses ONLY .transform() — never .fit() or .fit_transform()
#   - Feature columns are added as NaN if missing from request
#   - Column order is ALWAYS reindexed to training order before transform
#   - Dtypes are coerced explicitly before each transform step
# =============================================================================

import os
import logging
import traceback
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
import joblib
import uvicorn

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Using DEBUG level so every preprocessing step is visible in logs.
# In production you can set LOG_LEVEL=INFO via env variable.
logging.basicConfig(
    level=logging.getLevelName(os.getenv("LOG_LEVEL", "DEBUG")),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sepsis_api")

# =============================================================================
# PATHS — Change MODEL_DIR to wherever your .pkl files live
# =============================================================================
MODEL_DIR       = os.getenv("MODEL_DIR", "models")
MODEL_PATH      = os.path.join(MODEL_DIR, "xgb_sepsis_model_final.pkl")
PREPRO_PATH     = os.path.join(MODEL_DIR, "sepsis_preprocessors_final.pkl")

# =============================================================================
# GLOBAL MODEL STORE — populated at startup
# =============================================================================
# Storing as a dict so the lifespan context manager can mutate it.
# Avoids global variable reassignment pitfalls.
model_store: Dict[str, Any] = {}


# =============================================================================
# LIFESPAN — Load model + preprocessors at startup, release at shutdown
# =============================================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs ONCE when the server starts.
    Loads the pickled model and all preprocessors into memory.
    This ensures zero file I/O during prediction requests.
    """
    logger.info("=== STARTUP: Loading model and preprocessors ===")
    logger.info(f"Model path      : {MODEL_PATH}")
    logger.info(f"Preprocessor    : {PREPRO_PATH}")

    # --- Validate files exist before loading ---
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            f"Place xgb_sepsis_model_final.pkl inside the '{MODEL_DIR}/' folder."
        )
    if not os.path.exists(PREPRO_PATH):
        raise FileNotFoundError(
            f"Preprocessors file not found: {PREPRO_PATH}\n"
            f"Place sepsis_preprocessors_final.pkl inside the '{MODEL_DIR}/' folder."
        )

    # --- Load ---
    model  = joblib.load(MODEL_PATH)
    prepro = joblib.load(PREPRO_PATH)

    # --- Extract and validate preprocessor contents ---
    # These MUST match exactly what was saved during training.
    required_keys = [
        "features", "numeric_cols", "cat_cols",
        "num_imputer", "scaler", "cat_imputer", "ord_encoder"
    ]
    for key in required_keys:
        if key not in prepro:
            raise KeyError(
                f"Preprocessor dict is missing key: '{key}'. "
                f"Available keys: {list(prepro.keys())}"
            )

    # --- Store in global dict ---
    model_store["model"]       = model
    model_store["features"]    = prepro["features"]       # Ordered list of all feature names
    model_store["numeric_cols"]= prepro["numeric_cols"]   # Subset: numeric features
    model_store["cat_cols"]    = prepro["cat_cols"]       # Subset: categorical features
    model_store["num_imputer"] = prepro["num_imputer"]    # sklearn SimpleImputer (median)
    model_store["scaler"]      = prepro["scaler"]         # sklearn StandardScaler
    model_store["cat_imputer"] = prepro["cat_imputer"]    # sklearn SimpleImputer (most_frequent) or None
    model_store["ord_encoder"] = prepro["ord_encoder"]    # sklearn OrdinalEncoder or None
    model_store["target"]      = prepro.get("target", "enr_cult_org_PATHO")

    logger.info(f"✅ Model loaded     : {type(model).__name__}")
    logger.info(f"✅ Total features   : {len(model_store['features'])}")
    logger.info(f"✅ Numeric features : {len(model_store['numeric_cols'])}")
    logger.info(f"✅ Cat features     : {len(model_store['cat_cols'])}")
    logger.info(f"✅ Target column    : {model_store['target']}")
    logger.info(f"✅ Feature list     : {model_store['features']}")

    yield  # Server is live; requests are served from here

    # --- Shutdown ---
    logger.info("=== SHUTDOWN: Clearing model store ===")
    model_store.clear()


# =============================================================================
# APP INITIALIZATION
# =============================================================================
app = FastAPI(
    title="Neonatal Sepsis Prediction API",
    description=(
        "Production XGBoost inference API for predicting neonatal sepsis probability. "
        "Send patient features as JSON; receive sepsis probability + risk category."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# =============================================================================
# CORS — Allow all origins for development.
# For production, replace ["*"] with your frontend URL list.
# =============================================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Change to ["https://yourfrontend.com"] in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# REQUEST / RESPONSE SCHEMAS
# =============================================================================

class PatientInput(BaseModel):
    """
    All 19 features used by the final trained model.
    Every field is Optional[float] so that:
      - Missing fields are automatically treated as NaN
      - The preprocessor's saved imputer fills them with training-set medians
    Never add required fields here — always accept partial input.
    """
    # Demographic
    ga:           Optional[float] = Field(None, description="Gestational age in weeks")
    bw:           Optional[float] = Field(None, description="Birth weight in grams")
    age_onset:    Optional[float] = Field(None, description="Age at onset of symptoms (hours/days)")

    # Clinical signs — binary (0/1) or numeric
    enr_leth:     Optional[float] = Field(None, description="Lethargy (0=No, 1=Yes)")
    enr_cry:      Optional[float] = Field(None, description="Abnormal cry (0=No, 1=Yes)")
    enr_refl:     Optional[float] = Field(None, description="Poor reflexes (0=No, 1=Yes)")
    enr_fever:    Optional[float] = Field(None, description="Fever (0=No, 1=Yes)")
    enr_tachyc:   Optional[float] = Field(None, description="Tachycardia (0=No, 1=Yes)")
    enr_tachyp:   Optional[float] = Field(None, description="Tachypnoea (0=No, 1=Yes)")
    enr_apn:      Optional[float] = Field(None, description="Apnoea (0=No, 1=Yes)")
    enr_retr:     Optional[float] = Field(None, description="Retractions (0=No, 1=Yes)")
    enr_cyan:     Optional[float] = Field(None, description="Cyanosis (0=No, 1=Yes)")
    enr_abd:      Optional[float] = Field(None, description="Abdominal distension (0=No, 1=Yes)")
    enr_puls:     Optional[float] = Field(None, description="Abnormal pulse (0=No, 1=Yes)")
    enr_hi_cry:   Optional[float] = Field(None, description="High-pitched cry (0=No, 1=Yes)")
    enr_cxr:      Optional[float] = Field(None, description="Abnormal CXR (0=No, 1=Yes)")
    enr_fio2:     Optional[float] = Field(None, description="FiO2 requirement (%)")

    # Lab values
    enr_crp_val:  Optional[float] = Field(None, description="CRP value (mg/dL)")
    enr_tlc_val:  Optional[float] = Field(None, description="TLC value (×10⁹/L)")

    class Config:
        # Allow extra fields — they are silently ignored.
        # This prevents crashes if the caller sends extra keys.
        extra = "ignore"


class PredictionResponse(BaseModel):
    """Response schema for a successful prediction."""
    sepsis_probability:    float  = Field(..., description="Probability of sepsis (0.0 – 1.0)")
    sepsis_probability_pct:str    = Field(..., description="Human-readable percentage")
    risk_category:         str    = Field(..., description="Low / Moderate / High / Very High")
    risk_message:          str    = Field(..., description="Clinical recommendation")
    prediction:            int    = Field(..., description="Binary prediction (0=No sepsis, 1=Sepsis)")
    features_received:     int    = Field(..., description="Number of non-null features in request")
    features_missing:      int    = Field(..., description="Features filled by imputer")
    features_used:         list   = Field(..., description="Exact feature list used by model")
    warning:               Optional[str] = Field(None, description="Any data quality warning")


class HealthResponse(BaseModel):
    status:     str
    model_loaded: bool
    features:   list
    n_features: int
    target:     str
    version:    str


# =============================================================================
# PREPROCESSING FUNCTION
# =============================================================================

def run_inference_preprocessing(patient_dict: dict) -> np.ndarray:
    """
    Exactly replicates training preprocessing in correct order:

    Step 1 — Build DataFrame from request dict
    Step 2 — Add ALL training features (missing ones become NaN)
    Step 3 — Reindex to EXACT training feature order
    Step 4 — Coerce numeric columns to float (prevent dtype mismatch)
    Step 5 — Apply num_imputer.transform() on numeric cols
    Step 6 — Apply cat_imputer.transform() on cat cols (if any)
    Step 7 — Apply ord_encoder.transform() on cat cols (if any)
    Step 8 — Apply scaler.transform() on numeric cols
    Step 9 — Recombine numeric + categorical in training order
    Step 10 — Return final numpy array with correct shape (1, n_features)

    NEVER calls .fit() or .fit_transform() — only .transform().
    """
    features    = model_store["features"]
    numeric_cols= model_store["numeric_cols"]
    cat_cols    = model_store["cat_cols"]
    num_imputer = model_store["num_imputer"]
    scaler      = model_store["scaler"]
    cat_imputer = model_store["cat_imputer"]
    ord_encoder = model_store["ord_encoder"]

    # ------------------------------------------------------------------
    # STEP 1: Build single-row DataFrame from the incoming request dict
    # ------------------------------------------------------------------
    logger.debug(f"[STEP 1] Raw patient_dict keys: {list(patient_dict.keys())}")
    df = pd.DataFrame([patient_dict])

    # ------------------------------------------------------------------
    # STEP 2: Add ANY missing training feature as NaN
    # This is the key fix for "feature mismatch" errors.
    # If the caller doesn't send enr_fio2, we add it as NaN so the
    # imputer can fill it with the training-set median.
    # ------------------------------------------------------------------
    for col in features:
        if col not in df.columns:
            df[col] = np.nan
            logger.debug(f"[STEP 2] Added missing feature '{col}' as NaN")

    # ------------------------------------------------------------------
    # STEP 3: Reindex to EXACT training order
    # XGBoost is strict about feature order — must match booster.feature_names
    # ------------------------------------------------------------------
    df = df.reindex(columns=features)
    logger.debug(f"[STEP 3] DataFrame shape after reindex: {df.shape}")
    logger.debug(f"[STEP 3] Columns: {df.columns.tolist()}")

    # ------------------------------------------------------------------
    # STEP 4: Coerce all numeric columns to float
    # Prevents dtype errors when a string like "3" is passed as a value
    # ------------------------------------------------------------------
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    logger.debug(f"[STEP 4] dtypes after coerce:\n{df[numeric_cols].dtypes.to_dict()}")

    # ------------------------------------------------------------------
    # STEP 5: Apply numeric imputer — fills NaN with training medians
    # CRITICAL: use .transform(), NOT .fit_transform()
    # ------------------------------------------------------------------
    if numeric_cols:
        df_num_values = df[numeric_cols].values.astype(np.float64)
        imputed_num   = num_imputer.transform(df_num_values)
        df_num        = pd.DataFrame(imputed_num, columns=numeric_cols)
        logger.debug(f"[STEP 5] Numeric imputation complete. Shape: {df_num.shape}")
    else:
        df_num = pd.DataFrame()

    # ------------------------------------------------------------------
    # STEP 6 + 7: Categorical imputer + OrdinalEncoder (if any cat cols)
    # In the final model (Case 6) there are NO categorical columns,
    # so this block is a safe no-op. Kept for forward compatibility.
    # ------------------------------------------------------------------
    if cat_cols and cat_imputer is not None and ord_encoder is not None:
        df_cat_raw = df[cat_cols].astype(object)
        df_cat_imputed = cat_imputer.transform(df_cat_raw.values)
        df_cat_encoded = ord_encoder.transform(df_cat_imputed)
        df_cat = pd.DataFrame(df_cat_encoded, columns=cat_cols)
        logger.debug(f"[STEP 6+7] Cat impute+encode complete. Shape: {df_cat.shape}")
    else:
        df_cat = pd.DataFrame()
        logger.debug("[STEP 6+7] No categorical columns — skipped.")

    # ------------------------------------------------------------------
    # STEP 8: Scale numeric columns
    # CRITICAL: use .transform(), NOT .fit_transform()
    # ------------------------------------------------------------------
    if not df_num.empty:
        scaled_num   = scaler.transform(df_num.values.astype(np.float64))
        df_num_scaled= pd.DataFrame(scaled_num, columns=numeric_cols)
        logger.debug(f"[STEP 8] Scaling complete. Shape: {df_num_scaled.shape}")
    else:
        df_num_scaled = pd.DataFrame()

    # ------------------------------------------------------------------
    # STEP 9: Recombine numeric + categorical, reorder to training order
    # This guarantees the final array column order == training order.
    # ------------------------------------------------------------------
    if not df_cat.empty:
        df_final = pd.concat([df_num_scaled, df_cat], axis=1)
    else:
        df_final = df_num_scaled.copy()

    # Always reindex to enforce training column order
    df_final = df_final.reindex(columns=features)
    logger.debug(f"[STEP 9] Final feature DataFrame shape: {df_final.shape}")

    # ------------------------------------------------------------------
    # STEP 10: Convert to numpy float32 array (what XGBoost expects)
    # ------------------------------------------------------------------
    X = df_final.values.astype(np.float32)

    # Final sanity check
    expected_n = len(features)
    actual_n   = X.shape[1]
    if actual_n != expected_n:
        raise ValueError(
            f"Feature count mismatch! Model expects {expected_n} features "
            f"but preprocessed array has {actual_n} features. "
            f"Expected: {features}"
        )

    logger.debug(f"[STEP 10] Final X shape: {X.shape}")
    return X


# =============================================================================
# RISK CATEGORY LOGIC
# =============================================================================

def get_risk_category(prob: float) -> tuple[str, str]:
    """
    Maps sepsis probability to a clinical risk category + recommendation.
    Thresholds are clinically motivated — adjust for your institution.
    """
    if prob < 0.20:
        return "Low",       "Routine monitoring. No immediate intervention required."
    elif prob < 0.50:
        return "Moderate",  "Close observation advised. Consider repeat labs in 6–12h."
    elif prob < 0.75:
        return "High",      "Immediate clinical evaluation required. Consider empirical antibiotics."
    else:
        return "Very High", "URGENT: High probability of sepsis. Initiate sepsis bundle immediately."


# =============================================================================
# ROUTES
# =============================================================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint — confirms API is running."""
    return {
        "message": "Neonatal Sepsis Prediction API is running.",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.
    Returns model load status, feature list, and target column.
    Use this to verify deployment before sending predictions.
    """
    model_loaded = "model" in model_store and model_store["model"] is not None

    return HealthResponse(
        status      = "healthy" if model_loaded else "degraded",
        model_loaded= model_loaded,
        features    = model_store.get("features", []),
        n_features  = len(model_store.get("features", [])),
        target      = model_store.get("target", "unknown"),
        version     = "1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientInput):
    """
    Main prediction endpoint.

    - Send any subset of the 19 features.
    - Missing features are automatically imputed using training-set medians.
    - Returns: sepsis_probability, risk_category, clinical recommendation.

    Example request body (minimal — only CRP + TLC):
    {
        "enr_crp_val": 12.5,
        "enr_tlc_val": 18.0
    }
    """
    logger.info("=== New prediction request received ===")

    # Guard: model must be loaded
    if "model" not in model_store:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Server is still starting up or failed to load."
        )

    # --- Convert Pydantic model to plain dict ---
    # exclude_none=False because we WANT None values (they become NaN)
    patient_dict = patient.dict()
    logger.debug(f"Patient dict (raw): {patient_dict}")

    # --- Count received vs missing features ---
    features         = model_store["features"]
    received_values  = {k: v for k, v in patient_dict.items() if v is not None and k in features}
    n_received       = len(received_values)
    n_missing        = len(features) - n_received

    logger.info(f"Features received : {n_received}/{len(features)}")
    logger.info(f"Features missing  : {n_missing}/{len(features)} (will be imputed)")

    if n_missing > 0:
        missing_names = [f for f in features if patient_dict.get(f) is None]
        logger.debug(f"Missing features  : {missing_names}")

    # --- Build warning if many features are missing ---
    warning = None
    if n_missing > len(features) * 0.5:
        warning = (
            f"{n_missing}/{len(features)} features are missing. "
            "Prediction reliability may be reduced. Provide more features for better accuracy."
        )
        logger.warning(warning)

    # --- Run preprocessing pipeline ---
    try:
        X = run_inference_preprocessing(patient_dict)
        logger.info(f"Preprocessing complete. Array shape: {X.shape}")
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=422,
            detail=f"Preprocessing error: {str(e)}"
        )

    # --- Model inference ---
    try:
        model = model_store["model"]
        proba = model.predict_proba(X)         # shape: (1, 2)
        prob_sepsis = float(proba[0, 1])       # probability of class 1 (sepsis)
        prediction  = int(prob_sepsis >= 0.5)  # binary threshold
        logger.info(f"Prediction: prob={prob_sepsis:.4f}, class={prediction}")
    except Exception as e:
        logger.error(f"Model inference failed: {e}\n{traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Model inference error: {str(e)}"
        )

    # --- Risk categorization ---
    risk_category, risk_message = get_risk_category(prob_sepsis)
    logger.info(f"Risk category: {risk_category}")

    return PredictionResponse(
        sepsis_probability    = round(prob_sepsis, 4),
        sepsis_probability_pct= f"{prob_sepsis * 100:.1f}%",
        risk_category         = risk_category,
        risk_message          = risk_message,
        prediction            = prediction,
        features_received     = n_received,
        features_missing      = n_missing,
        features_used         = features,
        warning               = warning,
    )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(patients: list[PatientInput]):
    """
    Batch prediction endpoint — send a list of patients, get back predictions.
    Maximum 100 patients per request (configurable via MAX_BATCH env var).
    """
    max_batch = int(os.getenv("MAX_BATCH", 100))

    if len(patients) > max_batch:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size {len(patients)} exceeds maximum {max_batch}."
        )

    if "model" not in model_store:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    logger.info(f"=== Batch prediction: {len(patients)} patients ===")
    results = []

    for i, patient in enumerate(patients):
        patient_dict = patient.dict()
        try:
            X = run_inference_preprocessing(patient_dict)
            proba = model_store["model"].predict_proba(X)
            prob  = float(proba[0, 1])
            risk, msg = get_risk_category(prob)
            results.append({
                "index":               i,
                "sepsis_probability":  round(prob, 4),
                "risk_category":       risk,
                "risk_message":        msg,
                "prediction":          int(prob >= 0.5),
                "error":               None,
            })
        except Exception as e:
            logger.error(f"Batch item {i} failed: {e}")
            results.append({
                "index":    i,
                "error":    str(e),
                "sepsis_probability": None,
                "risk_category":      "Error",
                "risk_message":       "Preprocessing or inference failed.",
                "prediction":         None,
            })

    return {"total": len(patients), "predictions": results}


# =============================================================================
# GLOBAL EXCEPTION HANDLER
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catches any unhandled exception and returns a clean JSON error."""
    logger.error(f"Unhandled exception on {request.url}: {exc}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={
            "error":   "Internal server error",
            "detail":  str(exc),
            "path":    str(request.url),
        }
    )


# =============================================================================
# LOCAL DEV ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True,          # Turn off reload=False in production
        log_level="debug",
    )
