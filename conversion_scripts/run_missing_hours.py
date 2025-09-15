#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Repair missing OpenVisus IDX timesteps from a CSV list of missing hours.

Usage examples:
  # Fix all missing hours listed in missing_hours.csv (write ONLY the missing hours per day)
  python repair_missing_hours.py --csv missing_hours.csv \
      --base-dir /glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc \
      --out-dir  /glade/work/dpanta/era5/idx/2T \
      --idx      /glade/work/dpanta/era5/idx/2T/era5_sfc_2T_zip.idx

  # Same, but overwrite entire days for each date that has any missing hour(s)
  python repair_missing_hours.py --csv missing_hours.csv --full-day ...

  # Limit to a date window
  python repair_missing_hours.py --csv missing_hours.csv --start 1969-12-01 --end 1970-01-31 ...

Notes:
- Assumes your absolute 1-based hour index scheme anchored at BASE_YEAR=1940 (same as your audit script).
- Handles partial-day gaps efficiently by writing only the missing hours (default).
"""

import os, re, glob, argparse, time, csv, sys
from collections import defaultdict
from datetime import datetime, date
import numpy as np
import xarray as xr

import OpenVisus as ov

# ----------------------------
# CONFIG DEFAULTS (edit if needed)
# ----------------------------
VAR_NAME   = "VAR_2T"     # name in NetCDF
FIELD_NAME = "2T"         # name in IDX
ARCO       = "2mb"
BASE_YEAR  = 1940         # must match your audit script

# ERA5 monthly file name pattern:
# e5.oper.an.sfc.128_167_2t.ll025sc.YYYYMMDDHH_YYYYMMDDHH.nc
FILE_REGEX = re.compile(
    r"e5\.oper\.an\.sfc\.128_167_2t\.ll025sc\.(\d{10})_(\d{10})\.nc$"
)

# ----------------------------
# Helpers for absolute hour index
# ----------------------------

def abs_time_index_hours_leapaware(ts_array: np.ndarray) -> np.ndarray:
    """
    Convert np.datetime64 array to absolute hourly index (1-based), leap-aware.
    """
    ts = ts_array.astype("datetime64[ns]")
    base = np.datetime64(f"{BASE_YEAR:04d}-01-01T00:00:00").astype("datetime64[ns]")
    hours_since = ((ts - base) / np.timedelta64(1, "h")).astype(np.int64)
    return hours_since + (BASE_YEAR * 365 * 24 + 1)

def hour_index_from_iso(iso: str) -> int:
    """ISO 'YYYY-MM-DD HH:MM:SS' -> absolute 1-based hour index (leap-aware)."""
    iso = iso.strip()
    if len(iso) <= 10:
        iso += " 00:00:00"
    dt = np.datetime64(iso.replace(" ", "T"))
    base0 = np.datetime64(f"{BASE_YEAR:04d}-01-01T00:00:00")
    hours_since = int((dt - base0) / np.timedelta64(1, "h"))
    return BASE_YEAR * 365 * 24 + hours_since + 1

# ----------------------------
# ERA5 dataset helpers
# ----------------------------

def find_month_file(base_dir: str, year: int, month: int) -> str:
    """
    ERA5 stores one big file per month. We locate it by YYYYMM directory and pick
    the file matching the 2T pattern.
    """
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
    # Usually there’s exactly one; if multiple, take the first (adjust if needed)
    return candidates[0]

def ensure_idx(idx_name: str, out_dir: str, nc_sample_path: str) -> ov.Dataset:
    """
    Create the IDX if it doesn't exist; otherwise open it.
    - Dims are set to (X, Y) and we write slices shaped (X, Y).
    - Time window is generous so any hour index fits.
    """
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(idx_name):
        return ov.LoadDataset(idx_name)

    # Sample dims/dtype/range from the provided NetCDF sample
    with xr.open_dataset(nc_sample_path) as ds:
        if VAR_NAME not in ds:
            raise RuntimeError(f"Variable '{VAR_NAME}' not found in {nc_sample_path}")
        sample = ds[VAR_NAME].isel(time=0).values  # (Y, X)
        if sample.ndim != 2:
            raise RuntimeError(f"{VAR_NAME} expected 2D slice (y,x); got {sample.shape}")
        Y, X = sample.shape
        dtype = sample.dtype
        vmin = np.nanmin(ds[VAR_NAME].values)
        vmax = np.nanmax(ds[VAR_NAME].values)

    fld = ov.Field.fromString(f"""{FIELD_NAME} {str(dtype)} format(row_major) min({vmin}) max({vmax})""")

    # Time window guard (covers well past your data; adjust if you like)
    TIME_START = BASE_YEAR * 365 * 24 + 1
    TIME_END   = (2101) * 366 * 24  # generous

    print(f"[INFO] Creating IDX at: {idx_name}")
    return ov.CreateIdx(
        url=idx_name,
        dims=[X, Y],            # NOTE: (X, Y) order
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
                           compress_per_hour: bool = True):
    """
    Write either ONLY the given missing hours (0..23) for a day, or all 24 if missing_hours_for_day is None.
    """
    # 1) Find the monthly file containing that day
    nc_path = find_month_file(base_dir, year, month)
    print(f"[INFO] Using monthly file: {nc_path}")

    # 2) Open once, select just the target day
    with xr.open_dataset(nc_path) as ds:
        if VAR_NAME not in ds:
            raise RuntimeError(f"Variable '{VAR_NAME}' not found in {nc_path}")
        if "time" not in ds:
            raise RuntimeError(f"No 'time' coordinate in {nc_path}")

        times_all = ds["time"].values  # datetime64 array
        # Build mask for the given day (00:00..23:00)
        day_start = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T00:00:00")
        day_end   = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T23:00:00")
        mask_day = (times_all >= day_start) & (times_all <= day_end)

        if not np.any(mask_day):
            raise RuntimeError(f"No hourly timestamps found in file for {year:04d}-{month:02d}-{day:02d}")

        times_day = times_all[mask_day]                  # (H,)
        data_day  = ds[VAR_NAME].values[mask_day, :, :]  # (H, Y, X)
        assert data_day.ndim == 3

        # If only specific hours are missing, filter to just those hours
        if missing_hours_for_day is not None:
            # Map np.datetime64 to hour-of-day
            hours_of_day = np.array([int(str(t)[11:13]) for t in times_day], dtype=int)
            select_mask = np.array([h in missing_hours_for_day for h in hours_of_day], dtype=bool)

            if not np.any(select_mask):
                print(f"[SKIP] No matching missing hours present in file for {year:04d}-{month:02d}-{day:02d}")
                return 0

            times_day = times_day[select_mask]
            data_day  = data_day[select_mask, :, :]

        # 3) Ensure IDX exists / open it
        db_idx = ensure_idx(idx_name, out_dir, nc_path)

        # 4) Compute absolute hour indices (1-based) and write
        t_abs = abs_time_index_hours_leapaware(times_day)

        print(f"[INFO] Writing {data_day.shape[0]} hour(s) for {year:04d}-{month:02d}-{day:02d}")
        t0 = time.time()
        written = 0
        for i in range(data_day.shape[0]):
            # Data layout: (Y, X) → IDX expects (X, Y) row-major; we can pass as-is since OpenVisus
            # uses row_major with dims=[X,Y] and NumPy array is row-major (Y rows, X cols) -> it matches.
            slice_yx = data_day[i, :, :]  # (Y, X)
            db_idx.write(slice_yx, time=int(t_abs[i]))
            if compress_per_hour:
                db_idx.compressDataset(["zip"], timestep=int(t_abs[i]))
            written += 1
        t1 = time.time()
        print(f"[INFO] Wrote {written} hour(s) in {t1 - t0:.2f} s")

        # Optionally, if not compressing per hour, compress once after the loop (here we keep per-hour default)

        return written

# ----------------------------
# CSV ingestion
# ----------------------------

def parse_missing_csv(csv_path: str,
                      start_date: str | None,
                      end_date: str | None) -> dict[date, set[int]]:
    """
    Read missing_hours.csv and return {date -> set(hours)}.
    CSV format:
      hour_index,iso_utc
      17257057,1969-12-18 00:00:00
      ...
    Optional inclusive date filters (YYYY-MM-DD).
    """
    d0 = datetime.strptime(start_date, "%Y-%m-%d").date() if start_date else None
    d1 = datetime.strptime(end_date,   "%Y-%m-%d").date() if end_date   else None

    per_day: dict[date, set[int]] = defaultdict(set)

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

    return per_day

# ----------------------------
# CLI
# ----------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Repair missing OpenVisus IDX timesteps from missing_hours.csv."
    )
    ap.add_argument("--csv", required=True,
                    help="Path to missing_hours.csv (columns: hour_index,iso_utc).")
    ap.add_argument("--base-dir", required=True,
                    help="Root ERA5 base dir (YYYYMM subdirs).")
    ap.add_argument("--out-dir",  required=True,
                    help="Output IDX directory.")
    ap.add_argument("--idx",      required=True,
                    help="IDX filename (e.g., /path/to/era5_sfc_2T_zip.idx).")
    ap.add_argument("--start", default=None,
                    help="(Optional) Start date inclusive (YYYY-MM-DD) to filter CSV.")
    ap.add_argument("--end",   default=None,
                    help="(Optional) End date inclusive (YYYY-MM-DD) to filter CSV.")
    ap.add_argument("--full-day", action="store_true",
                    help="Write all 24 hours for each affected date (instead of only missing hours).")
    ap.add_argument("--no-per-hour-compress", action="store_true",
                    help="Do not call compressDataset(['zip']) per written hour.")
    ap.add_argument("--limit-days", type=int, default=None,
                    help="Optional cap on number of dates to process this run.")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    idx_name = os.path.abspath(args.idx)
    out_dir  = os.path.abspath(args.out_dir)
    base_dir = os.path.abspath(args.base_dir)

    if not os.path.isfile(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    per_day = parse_missing_csv(csv_path, args.start, args.end)
    if not per_day:
        print("[OK] No rows matched the filter(s); nothing to do.")
        return

    # Sort dates ascending
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
        year, month, day = d.year, d.month, d.day
        hours = None if args.full_day else per_day[d]
        written = write_missing_for_date(
            base_dir=base_dir,
            idx_name=idx_name,
            out_dir=out_dir,
            year=year, month=month, day=day,
            missing_hours_for_day=hours,
            compress_per_hour=(not args.no_per_hour_compress),
        )
        total_written += int(written or 0)
        total_days    += 1

    # Summary (matches your audit style)
    print("\n[SUMMARY]")
    print(f"  Dates processed: {total_days}")
    print(f"  Total hours written this run: {total_written}")

if __name__ == "__main__":
    main()

