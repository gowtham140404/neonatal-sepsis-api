#!/usr/bin/env python3
"""
test_api.py — Local integration tests for the Neonatal Sepsis API

Run BEFORE deploying to catch any issues.
Usage:
    # 1. Start the API in one terminal:
    #    uvicorn main:app --reload
    #
    # 2. Run tests in another terminal:
    #    python test_api.py

All tests use the requests library (pip install requests).
"""

import sys
import json
import requests

BASE_URL = "http://localhost:8000"

PASS = "\033[92m✅ PASS\033[0m"
FAIL = "\033[91m❌ FAIL\033[0m"

results = []


def test(name, fn):
    """Run a single test and record result."""
    try:
        fn()
        print(f"{PASS} — {name}")
        results.append((name, True, None))
    except AssertionError as e:
        print(f"{FAIL} — {name}: {e}")
        results.append((name, False, str(e)))
    except Exception as e:
        print(f"{FAIL} — {name}: UNEXPECTED ERROR: {e}")
        results.append((name, False, str(e)))


# =============================================================================
# TEST 1: Root endpoint
# =============================================================================
def test_root():
    r = requests.get(f"{BASE_URL}/")
    assert r.status_code == 200, f"Status {r.status_code}"
    data = r.json()
    assert "message" in data, f"No 'message' key in {data}"


# =============================================================================
# TEST 2: Health check
# =============================================================================
def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Status {r.status_code}"
    data = r.json()
    assert data["model_loaded"] == True, f"Model not loaded: {data}"
    assert data["n_features"] == 19, f"Expected 19 features, got {data['n_features']}"
    assert data["status"] == "healthy", f"Status not healthy: {data}"
    print(f"       Features: {data['features']}")


# =============================================================================
# TEST 3: Full feature input (all 19 features provided)
# =============================================================================
def test_predict_full():
    payload = {
        "ga": 32.0,
        "bw": 1500.0,
        "enr_leth": 1.0,
        "enr_cry": 0.0,
        "enr_refl": 1.0,
        "enr_fever": 1.0,
        "enr_tachyc": 1.0,
        "enr_tachyp": 1.0,
        "enr_apn": 0.0,
        "enr_retr": 1.0,
        "enr_cyan": 0.0,
        "enr_abd": 1.0,
        "enr_puls": 0.0,
        "enr_hi_cry": 0.0,
        "enr_cxr": 1.0,
        "enr_fio2": 40.0,
        "enr_crp_val": 25.5,
        "enr_tlc_val": 22.0,
        "age_onset": 48.0,
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200, f"Status {r.status_code}, body: {r.text}"
    data = r.json()
    assert 0.0 <= data["sepsis_probability"] <= 1.0, f"Invalid prob: {data['sepsis_probability']}"
    assert data["risk_category"] in ["Low", "Moderate", "High", "Very High"]
    assert data["prediction"] in [0, 1]
    assert data["features_missing"] == 0
    print(f"       Probability: {data['sepsis_probability_pct']} | Risk: {data['risk_category']}")


# =============================================================================
# TEST 4: Partial input — only CRP + TLC (most common clinical scenario)
# =============================================================================
def test_predict_partial_crp_tlc():
    payload = {
        "enr_crp_val": 8.2,
        "enr_tlc_val": 15.0,
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200, f"Status {r.status_code}, body: {r.text}"
    data = r.json()
    assert 0.0 <= data["sepsis_probability"] <= 1.0
    assert data["features_missing"] == 17   # 19 - 2 = 17 imputed
    assert data["warning"] is not None       # Should warn about many missing
    print(f"       Probability: {data['sepsis_probability_pct']} | Missing: {data['features_missing']}")


# =============================================================================
# TEST 5: Minimal input — only gestational age + birth weight
# =============================================================================
def test_predict_minimal():
    payload = {
        "ga": 28.0,
        "bw": 900.0,
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200, f"Status {r.status_code}, body: {r.text}"
    data = r.json()
    assert 0.0 <= data["sepsis_probability"] <= 1.0
    print(f"       Probability: {data['sepsis_probability_pct']} | Risk: {data['risk_category']}")


# =============================================================================
# TEST 6: Empty input — all NaN (model uses all imputed medians)
# =============================================================================
def test_predict_empty():
    payload = {}
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200, f"Status {r.status_code}, body: {r.text}"
    data = r.json()
    assert 0.0 <= data["sepsis_probability"] <= 1.0
    assert data["features_missing"] == 19
    print(f"       Probability: {data['sepsis_probability_pct']} (all imputed)")


# =============================================================================
# TEST 7: Extra keys in request (should be silently ignored)
# =============================================================================
def test_predict_extra_keys():
    payload = {
        "enr_crp_val": 5.0,
        "enr_tlc_val": 10.0,
        "non_existent_column": "should_be_ignored",
        "another_random_key": 999,
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    assert r.status_code == 200, f"Status {r.status_code}, body: {r.text}"
    print(f"       Extra keys correctly ignored.")


# =============================================================================
# TEST 8: String numbers in request (should be coerced to float)
# =============================================================================
def test_predict_string_numbers():
    payload = {
        "enr_crp_val": "12.5",   # String, not float
        "enr_tlc_val": "18",      # String integer
        "ga": "32",
    }
    r = requests.post(f"{BASE_URL}/predict", json=payload)
    # Pydantic will coerce strings to float for Optional[float] fields
    # If Pydantic rejects, 422 is acceptable here
    assert r.status_code in [200, 422], f"Unexpected status: {r.status_code}"
    if r.status_code == 200:
        print(f"       String numbers coerced correctly.")
    else:
        print(f"       String numbers rejected by Pydantic (422 — acceptable).")


# =============================================================================
# TEST 9: Batch prediction
# =============================================================================
def test_predict_batch():
    payload = [
        {"ga": 30.0, "bw": 1200.0, "enr_crp_val": 20.0, "enr_tlc_val": 25.0},
        {"enr_crp_val": 2.0, "enr_tlc_val": 8.0},
        {},  # All NaN
    ]
    r = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    assert r.status_code == 200, f"Status {r.status_code}, body: {r.text}"
    data = r.json()
    assert data["total"] == 3
    assert len(data["predictions"]) == 3
    for pred in data["predictions"]:
        if pred["error"] is None:
            assert 0.0 <= pred["sepsis_probability"] <= 1.0
    print(f"       Batch of 3 — all returned successfully.")


# =============================================================================
# TEST 10: Invalid endpoint (404)
# =============================================================================
def test_404():
    r = requests.get(f"{BASE_URL}/nonexistent_route")
    assert r.status_code == 404, f"Expected 404, got {r.status_code}"


# =============================================================================
# RUN ALL TESTS
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  NEONATAL SEPSIS API — LOCAL TEST SUITE")
    print("=" * 60 + "\n")

    test("Root endpoint",                  test_root)
    test("Health check",                   test_health)
    test("Predict — full 19 features",     test_predict_full)
    test("Predict — partial (CRP + TLC)",  test_predict_partial_crp_tlc)
    test("Predict — minimal (ga + bw)",    test_predict_minimal)
    test("Predict — empty input (all NaN)",test_predict_empty)
    test("Predict — extra keys ignored",   test_predict_extra_keys)
    test("Predict — string numbers",       test_predict_string_numbers)
    test("Predict — batch endpoint",       test_predict_batch)
    test("404 handling",                   test_404)

    print("\n" + "=" * 60)
    passed = sum(1 for _, ok, _ in results if ok)
    total  = len(results)
    print(f"  RESULTS: {passed}/{total} passed")
    if passed < total:
        print("\n  FAILURES:")
        for name, ok, err in results:
            if not ok:
                print(f"    ❌ {name}: {err}")
    else:
        print("  🎉 All tests passed! Safe to deploy.")
    print("=" * 60 + "\n")

    sys.exit(0 if passed == total else 1)
