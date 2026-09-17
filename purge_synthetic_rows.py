"""
purge_synthetic_rows.py

Identifies synthetic backfill rows in the Supabase `aqi_features` table
using the deterministic signature left by backfill_up3.py's
MONTHLY_AQI_BASELINES formula:

    pm10 == round(pm25 * 1.4, 1)

Real OpenWeatherMap-derived rows do not follow this exact fixed ratio,
so this signature reliably separates the two populations (confirmed
against the uploaded feature export: ~56% of all rows match it,
concentrated May 2025-May 2026, dropping to ~0% from June 2026 on).

Usage:
    python purge_synthetic_rows.py --dry-run     # inspect only, no changes
    python purge_synthetic_rows.py --tag-only    # add a data_source column value, keep all rows
    python purge_synthetic_rows.py --delete      # permanently remove synthetic rows (asks to confirm)

Recommended order:
    1. Run --dry-run first and sanity-check the counts/date range printed.
    2. Run --tag-only. This is reversible and lets training_pipeline.py
       filter on data_source without losing the synthetic rows for
       inspection/audit purposes.
    3. Once you're confident, optionally run --delete to shrink the table.

NOTE: --tag-only requires a `data_source` text column on aqi_features.
If it doesn't exist yet, run this in the Supabase SQL editor first:
    ALTER TABLE aqi_features ADD COLUMN IF NOT EXISTS data_source text;
"""
import os
import argparse
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit("SUPABASE_URL / SUPABASE_KEY not found. Check your .env file.")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
BATCH = 200


def load_all_rows() -> pd.DataFrame:
    """Paginate through aqi_features pulling only the columns we need."""
    all_rows = []
    page = 0
    while True:
        result = (
            supabase.table("aqi_features")
            .select("id,timestamp,pm25,pm10")
            .order("timestamp")
            .range(page * 1000, (page + 1) * 1000 - 1)
            .execute()
        )
        if not result.data:
            break
        all_rows.extend(result.data)
        if len(result.data) < 1000:
            break
        page += 1
    df = pd.DataFrame(all_rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def flag_synthetic(df: pd.DataFrame) -> pd.DataFrame:
    df["pm10_expected"] = (df["pm25"] * 1.4).round(1)
    df["is_synthetic"] = df["pm10"].round(1) == df["pm10_expected"]
    return df


def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i : i + n]


def main():
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="Inspect only, no changes")
    mode.add_argument("--tag-only", action="store_true", help="Tag rows via data_source column, keep all rows")
    mode.add_argument("--delete", action="store_true", help="Permanently delete synthetic rows")
    args = parser.parse_args()

    print("Loading rows from Supabase (this can take a minute for 10k+ rows)...")
    df = load_all_rows()
    df = flag_synthetic(df)

    n_total = len(df)
    n_synthetic = int(df["is_synthetic"].sum())
    n_real = n_total - n_synthetic

    print(f"\nTotal rows:      {n_total}")
    print(f"Synthetic rows:  {n_synthetic} ({100 * n_synthetic / n_total:.1f}%)")
    print(f"Real rows:       {n_real} ({100 * n_real / n_total:.1f}%)")

    synth = df[df["is_synthetic"]]
    real = df[~df["is_synthetic"]]
    if len(synth):
        print(f"\nSynthetic date range: {synth['timestamp'].min()} -> {synth['timestamp'].max()}")
    if len(real):
        print(f"Real date range:      {real['timestamp'].min()} -> {real['timestamp'].max()}")

    print("\nMonthly synthetic fraction:")
    monthly = df.set_index("timestamp").resample("ME")["is_synthetic"].mean()
    for month, frac in monthly.items():
        print(f"  {month.strftime('%Y-%m')}: {frac * 100:5.1f}% synthetic")

    if args.dry_run:
        print("\nDry run complete. No changes made.")
        return

    synthetic_ids = df.loc[df["is_synthetic"], "id"].tolist()

    if args.tag_only:
        print(f"\nTagging {len(synthetic_ids)} rows as data_source='synthetic_backfill'...")
        for batch in batched(synthetic_ids, BATCH):
            supabase.table("aqi_features").update(
                {"data_source": "synthetic_backfill"}
            ).in_("id", batch).execute()

        real_ids = df.loc[~df["is_synthetic"], "id"].tolist()
        print(f"Tagging {len(real_ids)} rows as data_source='openweather_live'...")
        for batch in batched(real_ids, BATCH):
            supabase.table("aqi_features").update(
                {"data_source": "openweather_live"}
            ).in_("id", batch).execute()

        print("\nTagging complete. Update training_pipeline.py's load_features() to filter:")
        print('    df = df[df["data_source"] != "synthetic_backfill"]')

    if args.delete:
        confirm = input(
            f"\nType DELETE to permanently remove {len(synthetic_ids)} synthetic rows: "
        )
        if confirm != "DELETE":
            print("Aborted — no rows deleted.")
            return
        print("Deleting...")
        for batch in batched(synthetic_ids, BATCH):
            supabase.table("aqi_features").delete().in_("id", batch).execute()
        print(f"Deleted {len(synthetic_ids)} synthetic rows.")


if __name__ == "__main__":
    main()
