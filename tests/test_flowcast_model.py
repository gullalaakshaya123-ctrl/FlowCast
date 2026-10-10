
from pathlib import Path
import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = PROJECT_ROOT / "models" / "xgboost_congestion_model.joblib"
DATA_FILE = PROJECT_ROOT / "data" / "interim" / "flowcast_features.csv"


def test_xgboost_model_exists():
    assert MODEL_FILE.exists(), f"Model not found: {MODEL_FILE}"


def test_xgboost_model_predicts_congestion():
    model = joblib.load(MODEL_FILE)
    data = pd.read_csv(DATA_FILE)

    features = list(model.feature_names_in_)
    input_data = data[features].dropna().head(1)

    assert not input_data.empty, "No complete input row is available"

    prediction = model.predict(input_data)

    assert len(prediction) == 1, "Model did not return one prediction"