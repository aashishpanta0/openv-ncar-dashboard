#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repair missing OpenVisus IDX timesteps from a CSV list of missing hours.

CSV format:
  hour_index,iso_utc
  17257057,1969-12-18 00:00:00
  ...

Examples:
  # Process only the missing hours per date (default)
  python repair_missing_hours.py --csv missing_hours.csv

  # Same, but force full-day rewrites whenever any hour is missing for that date
  python repair_missing_hours.py --csv missing_hours.csv --full-day

  # Filter a date window and pass an explicit sample NetCDF for first-time IDX creation
  python repair_missing_hours.py --csv missing_hours.csv \
      --start 1969-12-01 --end 1970-01-31 \
      --sample-nc /glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc/202506/e5.oper.an.sfc.128_167_2t.ll025sc.2025060100_2025063023.nc
"""

import os, re, glob, argparse, time, csv, sys
from collections import defaultdict
from datetime import datetime, date
import numpy as np
import xarray as xr
import OpenVisus as ov

# ----------------------------
# Defaults tailored to your environment
# ----------------------------
DEFAULT_BASE_DIR = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc"
DEFAULT_IDX      = "/glade/work/dpanta/era5/idx/2T/era5_sfc_2T_zip.idx"
DEFAULT_OUT_DIR  = os.path.dirname(DEFAULT_IDX)

# OpenVisus / field config
VAR_NAME   = "VAR_2T"  # NetCDF variable
FIELD_NAME = "2T"      # IDX field
ARCO       = "2mb"

# Indexing scheme baseline (must match your audit script)
BASE_YEAR  = 1940

# ERA5 monthly filename pattern
FILE_REGEX = re.compile(
    r"e5\.oper\.an\.sfc\.128_167_2t\.ll025sc\.(\d{10})_(\d{10})\.nc$"
)

# ----------------------------
# Absolute hour index helpers
# ----------------------------

def abs_time_index_hours_leapaware(ts_array: np.ndarray) -> np.ndarray:
    """np.datetime64 array -> absolute hourly index (1-based), leap-aware."""
    ts = ts_array.astype("datetime64[ns]")
    base = np.datetime64(f"{BASE_YEAR:04d}-01-01T00:00:00").astype("datetime64[ns]")
    hours_since = ((ts - base) / np.timedelta64(1, "h")).astype(np.int64)
    return hours_since + (BASE_YEAR * 365 * 24 + 1)

# ----------------------------
# ERA5 dataset helpers
# ----------------------------

def find_month_file(base_dir: str, year: int, month: int) -> str:
    """Locate the single ERA5 monthly file for 2T in base_dir/YYYYMM/."""
    yyyymm = f"{year:04d}{month:02d}"
    month_dir = os.path.join(base_dir, yyyymm)
    if not os.path.isdir(month_dir):
        raise FileNotFoundError(f"Month directory not found: {month_dir}")

    candidates = sorted(
        f for f in glob.glob(os.path.join(month_dir, "e5.oper.an.sfc.128_167_2t.ll025sc.*_*.nc"))
        if FILE_REGEX.search(os.path.basename(f))
    )
    if not candidates:
        raise FileNotFoundError(f"No ERA5 2T file found in {month_dir}")
    return candidates[0]

def ensure_idx(idx_name: str, out_dir: str, sample_nc_path: str) -> ov.Dataset:
    """
    Create the IDX if it doesn't exist; otherwise open it.
    - Dims set to (X, Y) from sample (NetCDF is [time,y,x] slices)
    - Generous time window (BASE_YEAR..2101) using time_%d/ directories
    """
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(idx_name):
        return ov.LoadDataset(idx_name)

    with xr.open_dataset(sample_nc_path) as ds:
        if VAR_NAME not in ds:
            raise RuntimeError(f"Variable '{VAR_NAME}' not found in {sample_nc_path}")
        # Use first available time slice to infer (Y,X)
        sample = ds[VAR_NAME].isel(time=0).values  # (Y, X)
        if sample.ndim != 2:
            raise RuntimeError(f"{VAR_NAME} expected 2D slice (y,x); got {sample.shape}")
        Y, X = sample.shape
        dtype = sample.dtype
        # Use dataset-wide min/max if inexpensive
        vmin = np.nanmin(ds[VAR_NAME].values)
        vmax = np.nanmax(ds[VAR_NAME].values)

    fld = ov.Field.fromString(f"""{FIELD_NAME} {str(dtype)} format(row_major) min({vmin}) max({vmax})""")

    TIME_START = BASE_YEAR * 365 * 24 + 1
    TIME_END   = (2101) * 366 * 24

    print(f"[INFO] Creating IDX at: {idx_name}")
    return ov.CreateIdx(
        url=idx_name,
        dims=[X, Y],            # OpenVisus expects (X, Y)
        fields=[fld],
        compression="raw",
        arco=ARCO,
        time=[TIME_START, TIME_END, "time_%d/"],
    )

# ----------------------------
# Writing routines
# ----------------------------

def write_missing_for_date(base_dir: str,
                           idx_name: str,
                           out_dir: str,
                           year: int,
                           month: int,
                           day: int,
                           missing_hours_for_day: set[int] | None,
                           sample_nc_hint: str | None,
                           compress_per_hour: bool = True) -> int:
    """
    Write either ONLY the given missing hours (0..23) for a day, or all 24 if missing_hours_for_day is None.
    Returns number of hours written.
    """
    nc_path = find_month_file(base_dir, year, month)
    print(f"[INFO] Using monthly file: {nc_path}")

    with xr.open_dataset(nc_path) as ds:
        if VAR_NAME not in ds:
            raise RuntimeError(f"Variable '{VAR_NAME}' not found in {nc_path}")
        if "time" not in ds:
            raise RuntimeError(f"No 'time' coordinate in {nc_path}")

        times_all = ds["time"].values  # datetime64[h]
        day_start = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T00:00:00")
        day_end   = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T23:00:00")
        mask_day  = (times_all >= day_start) & (times_all <= day_end)

        if not np.any(mask_day):
            print(f"[WARN] No hourly timestamps found for {year:04d}-{month:02d}-{day:02d} in {nc_path}")
            return 0

        times_day = times_all[mask_day]                  # (H,)
        data_day  = ds[VAR_NAME].values[mask_day, :, :]  # (H, Y, X)
        assert data_day.ndim == 3

        if missing_hours_for_day is not None:
            hours_of_day = np.array([int(str(t)[11:13]) for t in times_day], dtype=int)
            select_mask  = np.array([h in missing_hours_for_day for h in hours_of_day], dtype=bool)
            if not np.any(select_mask):
                print(f"[SKIP] No matching missing hours present in file for {year:04d}-{month:02d}-{day:02d}")
                return 0
            times_day = times_day[select_mask]
            data_day  = data_day[select_mask, :, :]

        # Ensure IDX exists / open it (use either hint or current nc as sample)
        sample_nc = sample_nc_hint if sample_nc_hint else nc_path
        db_idx = ensure_idx(idx_name, out_dir, sample_nc)

        # Compute absolute hour indices and write
        t_abs = abs_time_index_hours_leapaware(times_day)

        print(f"[INFO] Writing {data_day.shape[0]} hour(s) for {year:04d}-{month:02d}-{day:02d}")
        t0 = time.time()
        written = 0
        for i in range(data_day.shape[0]):
            slice_yx = data_day[i, :, :]  # (Y, X), row-major
            db_idx.write(slice_yx, time=int(t_abs[i]))
            if compress_per_hour:
                db_idx.compressDataset(["zip"], timestep=int(t_abs[i]))
            written += 1
        t1 = time.time()
        print(f"[INFO] Wrote {written} hour(s) in {t1 - t0:.2f} s")

        return written

# ----------------------------
# CSV ingestion
# ----------------------------

def parse_missing_csv(csv_path: str,
                      start_date: str | None,
                      end_date: str | None) -> tuple[dict[date, set[int]], int]:
    """
    Read missing_hours.csv and return ({date -> set(hours)}, total_rows_in_window).
    CSV columns: hour_index, iso_utc
    """
    d0 = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    d1 = datetime.strptime(end_date,   "%Y-%m-%d").date() if end_date   else None

    per_day: dict[date, set[int]] = defaultdict(set)
    total_rows = 0

    with open(csv_path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        if "iso_utc" not in rdr.fieldnames:
            raise RuntimeError("CSV must have a header with at least the 'iso_utc' column.")
        for row in rdr:
            iso = row["iso_utc"].strip()
            if not iso:
                continue
            dt = datetime.strptime(iso, "%Y-%m-%d %H:%M:%S")
            d  = dt.date()
            if (d0 and d < d0) or (d1 and d > d1):
                continue
            per_day[d].add(dt.hour)
            total_rows += 1

    return per_day, total_rows

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(description="Repair missing OpenVisus IDX timesteps from missing_hours.csv.")
    ap.add_argument("--csv", required=True, help="Path to missing_hours.csv (columns: hour_index,iso_utc).")
    ap.add_argument("--base-dir", default=DEFAULT_BASE_DIR, help=f"Root ERA5 base dir (default: {DEFAULT_BASE_DIR})")
    ap.add_argument("--out-dir",  default=DEFAULT_OUT_DIR,  help=f"Output IDX directory (default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--idx",      default=DEFAULT_IDX,      help=f"IDX filename (default: {DEFAULT_IDX})")
    ap.add_argument("--start", default=None, help="(Optional) Start date inclusive (YYYY-MM-DD) to filter CSV.")
    ap.add_argument("--end",   default=None, help="(Optional) End date inclusive (YYYY-MM-DD) to filter CSV.")
    ap.add_argument("--full-day", action="store_true",
                    help="Write all 24 hours for each affected date (instead of only missing hours).")
    ap.add_argument("--no-per-hour-compress", action="store_true",
                    help="Do not call compressDataset(['zip']) per written hour.")
    ap.add_argument("--limit-days", type=int, default=None,
                    help="Optional cap on number of dates to process this run.")
    ap.add_argument("--sample-nc", default=None,
                    help="Optional NetCDF path to use as the sample for initial IDX creation "
                         "(e.g., 202506 monthly file). If omitted, the script uses the monthly file of the first date.")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    idx_name = os.path.abspath(args.idx)
    out_dir  = os.path.abspath(args.out_dir)
    base_dir = os.path.abspath(args.base_dir)
    sample_nc = os.path.abspath(args.sample_nc) if args.sample_nc else None

    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    per_day, total_missing_rows = parse_missing_csv(csv_path, args.start, args.end)
    if not per_day:
        print("[OK] No rows matched the filter(s); nothing to do.")
        return

    # Sort dates ascending and optionally limit
    dates = sorted(per_day.keys())
    if args.limit_days is not None:
        dates = dates[:max(0, args.limit_days)]

    total_written = 0
    total_days    = 0

    print(f"[INFO] IDX:       {idx_name}")
    print(f"[INFO] OUT DIR:   {out_dir}")
    print(f"[INFO] ERA5 DIR:  {base_dir}")
    print(f"[INFO] Dates to process: {len(dates)} (full-day={args.full_day})")

    for d in dates:
        y, m, dd = d.year, d.month, d.day
        hours = None if args.full_day else per_day[d]
        written = write_missing_for_date(
            base_dir=base_dir,
            idx_name=idx_name,
            out_dir=out_dir,
            year=y, month=m, day=dd,
            missing_hours_for_day=hours,
            sample_nc_hint=sample_nc,
            compress_per_hour=(not args.no_per_hour_compress),
        )
        total_written += int(written or 0)
        total_days    += 1

    # Summaries
    print("\n[SUMMARY]")
    print(f"  Dates processed: {total_days}")
    print(f"  Total missing hours in CSV window: {total_missing_rows}")
    print(f"  Total hours written this run:      {total_written}")

if __name__ == "__main__":
    main()

