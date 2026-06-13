import os

# ── Base path = wherever this file lives ──────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))

# ── Your single CSV file ──────────────────────────────────────────
DATA_PATH  = os.path.join(BASE_DIR, "patient_data.csv")

# ── Auto-created folders ──────────────────────────────────────────
MODEL_DIR  = os.path.join(BASE_DIR, "models\\")
REPORT_DIR = os.path.join(BASE_DIR, "reports\\")

TARGET_COLUMN = "Risk_Label"
TEST_SIZE     = 0.2
RANDOM_STATE  = 42