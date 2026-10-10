
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "flowcast_ml_ready.csv"


def test_model_ready_dataset_exists():
    assert DATA_FILE.exists(), f"Dataset not found: {DATA_FILE}"


def test_model_ready_dataset_has_required_columns():
    df = pd.read_csv(DATA_FILE, nrows=5)

    required_columns = [
        "traffic_volume",
        "avg_speed",
        "occupancy",
        "road_capacity",
        "congestion_level",
    ]

    for column in required_columns:
        assert column in df.columns, f"Missing column: {column}"


def test_congestion_labels_are_valid():
    df = pd.read_csv(DATA_FILE, usecols=["congestion_level"])

    allowed_labels = {"Free-flow", "Moderate", "Heavy", "Severe"}
    actual_labels = set(df["congestion_level"].dropna().unique())

    assert actual_labels, "No congestion labels found"
    assert actual_labels.issubset(allowed_labels), (
        f"Unexpected congestion labels: {actual_labels - allowed_labels}"
    )