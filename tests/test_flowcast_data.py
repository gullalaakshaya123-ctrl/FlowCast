
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "raw" / "traffic_sensor_log.csv"


def test_raw_dataset_exists():
    assert DATA_FILE.exists(), f"Dataset not found: {DATA_FILE}"


def test_raw_dataset_has_required_columns():
    df = pd.read_csv(DATA_FILE, nrows=5)

    required_columns = {
        "traffic_volume",
        "avg_speed",
        "occupancy",
    }

    assert required_columns.issubset(df.columns)


def test_raw_dataset_is_not_empty():
    df = pd.read_csv(DATA_FILE, usecols=["traffic_volume"])
    assert not df.empty