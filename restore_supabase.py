"""
restore_supabase.py  (v3 - fixed integer casting)
Restores the Supabase `aqi_features` table from the GitHub CSV backup.
"""

import os, sys, math
import pandas as pd
import requests
from io import StringIO

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ Loaded .env file")
except ImportError:
    pass

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
TABLE_NAME   = "aqi_features"
BATCH_SIZE   = 500

CSV_URL = (
    "https://raw.githubusercontent.com/"
    "FaiqaRashid99/pearls-aqi-predictor/main/feature_store/aqi_features.csv"
)

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ SUPABASE_URL or SUPABASE_KEY not set.")
    sys.exit(1)

try:
    from supabase import create_client
except ImportError:
    print("❌ Run:  pip install supabase")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
print(f"✅ Connected to Supabase: {SUPABASE_URL}")

# ── Step 1: Download CSV ──────────────────────────────────────────────────────
print("\n📥 Downloading CSV from GitHub...")
response = requests.get(CSV_URL, timeout=30)
response.raise_for_status()
df = pd.read_csv(StringIO(response.text))
print(f"✅ Downloaded {len(df)} rows × {len(df.columns)} columns")
print(f"   Columns: {list(df.columns)}")

# ── Step 2: Fix column types to match Supabase schema ────────────────────────
print("🔧 Casting column types...")

# These columns should be INTEGER in Supabase
INTEGER_COLS = [
    'hour', 'day', 'month', 'dayofweek',
    'is_weekend', 'is_rush_hour',
    'weather_code', 'weather',          # if stored as int
]

# These should be FLOAT
FLOAT_COLS = [
    'aqi', 'pm25', 'pm10', 'no2', 'o3', 'co', 'so2',
    'temp', 'feels_like', 'humidity', 'pressure',
    'wind_speed', 'wind_direction', 'visibility',
    'precipitation', 'aqi_change_rate',
]

# TEXT columns — leave as-is
TEXT_COLS = ['timestamp', 'city', 'weather']

for col in INTEGER_COLS:
    if col in df.columns:
        # Convert float like 0.0 → 0, keeping NaN as NaN for now
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Round to avoid 0.9999 → 0 issues, then cast
        df[col] = df[col].round(0)

for col in FLOAT_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# ── Step 3: Sanitise every value (NaN/Inf → None, correct Python types) ──────
print("🧹 Sanitising values...")

def sanitise(col_name, val):
    if val is None:
        return None
    # Handle numpy types
    try:
        import numpy as np
        if isinstance(val, np.bool_):
            return bool(val)
        if isinstance(val, np.integer):
            return int(val)
        if isinstance(val, np.floating):
            val = float(val)
    except ImportError:
        pass
    # NaN / Inf check for floats
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        # If this column is meant to be integer, cast it
        if col_name in INTEGER_COLS:
            return int(round(val))
        return float(val)
    if isinstance(val, int):
        return val
    return val  # string / None

records = []
for row in df.to_dict(orient="records"):
    clean_row = {k: sanitise(k, v) for k, v in row.items()}
    records.append(clean_row)

# Verify
bad = [(k, v) for row in records for k, v in row.items()
       if isinstance(v, float) and (math.isnan(v) or math.isinf(v))]
print(f"✅ Sanitised — remaining bad values: {len(bad)}  (should be 0)")

# Show sample of first record so you can verify types
print(f"\n🔍 Sample record (first row):")
for k, v in records[0].items():
    print(f"   {k}: {repr(v)}  ({type(v).__name__})")

# ── Step 4: Delete existing rows ─────────────────────────────────────────────
if os.environ.get("SKIP_DELETE") != "1":
    print(f"\n🗑️  Deleting all existing rows from `{TABLE_NAME}`...")
    try:
        supabase.table(TABLE_NAME).delete().gte("aqi", -9999).execute()
        print("✅ Deleted existing rows")
    except Exception as e:
        print(f"❌ Delete failed: {e}")
        sys.exit(1)
else:
    print("⏭️  Skipping delete (SKIP_DELETE=1)")

# ── Step 5: Insert in batches ─────────────────────────────────────────────────
print(f"\n📤 Inserting {len(records)} rows in batches of {BATCH_SIZE}...")
total_inserted = 0
errors = []

for i in range(0, len(records), BATCH_SIZE):
    batch = records[i : i + BATCH_SIZE]
    batch_num = i // BATCH_SIZE + 1
    try:
        supabase.table(TABLE_NAME).insert(batch).execute()
        total_inserted += len(batch)
        print(f"   Batch {batch_num}: ✅ {len(batch)} rows  (total: {total_inserted})")
    except Exception as e:
        errors.append((batch_num, str(e)))
        print(f"   Batch {batch_num}: ❌ {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
if not errors:
    print(f"✅ RESTORE COMPLETE — {total_inserted}/{len(records)} rows inserted")
else:
    print(f"⚠️  RESTORE PARTIAL — {total_inserted}/{len(records)} rows inserted")
    for batch_num, err in errors:
        print(f"   Batch {batch_num}: {err}")
print(f"{'='*50}\n")