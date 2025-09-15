#!/usr/bin/env python3
import os, re, glob, argparse, time
import numpy as np
import xarray as xr

import OpenVisus as ov

# ----------------------------
# CONFIG DEFAULTS (edit if needed)
# ----------------------------
BASE_DIR = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc"
OUT_DIR  = "/glade/work/dpanta/era5/idx/2T"
IDX_NAME = os.path.join(OUT_DIR, "era5_sfc_2T_zip.idx")

VAR_NAME   = "VAR_2T"     # name in NetCDF
FIELD_NAME = "2T"         # name in IDX
ARCO       = "2mb"

# ERA5 monthly file name pattern:
# e5.oper.an.sfc.128_167_2t.ll025sc.YYYYMMDDHH_YYYYMMDDHH.nc
FILE_REGEX = re.compile(
    r"e5\.oper\.an\.sfc\.128_167_2t\.ll025sc\.(\d{10})_(\d{10})\.nc$"
)

BASE_YEAR = 1940

def calculate_abs_hour_index(year: int, month: int, day: int, hour: int) -> int:
    """
    Hour-based, 1-based absolute index consistent with your scheme:
      raw = BASE_YEAR*365*24 + days_since_BASE*24 + hour + 1
    Leap-aware via numpy datetime arithmetic.
    """
    base = np.datetime64(f"{BASE_YEAR:04d}-01-01T00:00:00")
    t    = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:00:00")
    days_since = int((t.astype("datetime64[D]") - base.astype("datetime64[D]")) / np.timedelta64(1, "D"))
    return BASE_YEAR * 365 * 24 + days_since * 24 + hour + 1

def abs_time_index_hours_leapaware(ts_array: np.ndarray) -> np.ndarray:
    """
    Convert np.datetime64 array to absolute hourly index (1-based), leap-aware.
    """
    ts = ts_array.astype("datetime64[ns]")
    base = np.datetime64(f"{BASE_YEAR:04d}-01-01T00:00:00").astype("datetime64[ns]")
    hours_since = ((ts - base) / np.timedelta64(1, "h")).astype(np.int64)
    return hours_since + (BASE_YEAR * 365 * 24 + 1)

def find_month_file(year: int, month: int) -> str:
    """
    ERA5 stores one big file per month. We locate it by YYYYMM directory and pick
    the file matching the 2T pattern.
    """
    yyyymm = f"{year:04d}{month:02d}"
    month_dir = os.path.join(BASE_DIR, yyyymm)
    if not os.path.isdir(month_dir):
        raise FileNotFoundError(f"Month directory not found: {month_dir}")

    candidates = sorted(
        f for f in glob.glob(os.path.join(month_dir, "e5.oper.an.sfc.128_167_2t.ll025sc.*_*.nc"))
        if FILE_REGEX.search(os.path.basename(f))
    )
    if not candidates:
        raise FileNotFoundError(f"No ERA5 2T file found in {month_dir}")
    # Usually there’s exactly one; if multiple, take the first (or add logic if needed)
    return candidates[0]

def ensure_idx(db_sample_path: str) -> ov.Dataset:
    """
    Create the IDX if it doesn't exist; otherwise open it.
    - Dims are set to (X, Y) and we write slices shaped (X, Y).
    - We store 'north-up' by flipping latitude once at write time and transposing (Y,X)->(X,Y).
    - Time window is a large guard so any hour index fits.
    """
    os.makedirs(OUT_DIR, exist_ok=True)
    if os.path.exists(IDX_NAME):
        return ov.LoadDataset(IDX_NAME)

    # Sample dims/dtype/range from the provided NetCDF sample
    with xr.open_dataset(db_sample_path) as ds:
        if VAR_NAME not in ds:
            raise RuntimeError(f"Variable '{VAR_NAME}' not found in {db_sample_path}")
        sample = ds[VAR_NAME].values  # (T, Y, X)
        if sample.ndim != 3:
            raise RuntimeError(f"{VAR_NAME} expected 3D (time,y,x); got {sample.shape}")
        _, Y, X = sample.shape
        dtype = sample.dtype
        vmin = np.nanmin(sample)
        vmax = np.nanmax(sample)

    fld = ov.Field.fromString(f"""{FIELD_NAME} {str(dtype)} format(row_major) min({vmin}) max({vmax})""")

    # Time window guard (covers well past your data; adjust if you like)
    TIME_START = BASE_YEAR * 365 * 24 + 1
    TIME_END   = (2031) * 366 * 24  # generous

    print(f"[INFO] Creating IDX at: {IDX_NAME}")
    return ov.CreateIdx(
        url=IDX_NAME,
        dims=[X, Y],            # NOTE: (X, Y) order
        fields=[fld],
        compression="raw",
        arco=ARCO,
        time=[TIME_START, TIME_END, "time_%d/"],
    )

def write_one_day(year: int, month: int, day: int):
    # 1) Find the monthly file containing that day
    nc_path = find_month_file(year, month)
    print(f"[INFO] Using monthly file: {nc_path}")

    # 2) Open once, select just the target day
    with xr.open_dataset(nc_path) as ds:
        if VAR_NAME not in ds:
            raise RuntimeError(f"Variable '{VAR_NAME}' not found in {nc_path}")
        if "time" not in ds:
            raise RuntimeError(f"No 'time' coordinate in {nc_path}")

        # Select all 24 hours for the target day
        # Many ERA5 files have hourly time stamps; we filter by date
        times = ds["time"].values  # datetime64 array
        # build a mask for the given day
        day_start = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T00:00:00")
        day_end   = np.datetime64(f"{year:04d}-{month:02d}-{day:02d}T23:00:00")
        # Because comparisons on numpy datetime64 are cheap, do range mask
        mask = (times >= day_start) & (times <= day_end)

        if not np.any(mask):
            raise RuntimeError(f"No hourly timestamps found in file for {year:04d}-{month:02d}-{day:02d}")

        arr_day   = ds[VAR_NAME].values[mask, :, :]  # (24, Y, X)
        times_day = ds["time"].values[mask]

        # 3) Ensure IDX exists / open it
        db_idx = ensure_idx(nc_path)

        # 4) Compute absolute hour indices (1-based) and write only these hours
        t_abs = abs_time_index_hours_leapaware(times_day)

        print(f"[INFO] Writing {arr_day.shape[0]} hours for {year:04d}-{month:02d}-{day:02d}")
        t0 = time.time()

        # We want north-up & (X,Y) to match dims=[X, Y]:
        #   Flip latitude (Y) once, then transpose (Y,X)->(X,Y)
        for i in range(arr_day.shape[0]):
            slice_xy  = arr_day[i, :, :]  # (X, Y), north-up
            db_idx.write(slice_xy, time=int(t_abs[i]))
            db_idx.compressDataset(["zip"],timestep=int(t_abs[i]))
        t1 = time.time()
        print(f"[INFO] Raw write complete in {t1 - t0:.2f} s")

        # 5) Compress (zip) the dataset to finalize
        print("[INFO] Compressing dataset with 'zip' ...")
        c0 = time.time()
        #db_idx.compressDataset(["zip"],time=t)
        c1 = time.time()
        print(f"[INFO] Compression done in {c1 - c0:.2f} s")
        print(f"[DONE] Wrote & compressed {year:04d}-{month:02d}-{day:02d}")

def main():
    global BASE_DIR, OUT_DIR, IDX_NAME
    ap = argparse.ArgumentParser(
        description="Convert a single ERA5 2T day (24 hours) to IDX and compress."
    )
    ap.add_argument("--date", required=True, help="Target day in YYYY-MM-DD (e.g., 1940-01-03)")
    ap.add_argument("--base-dir", default=BASE_DIR, help="Root ERA5 base dir (default: set in script)")
    ap.add_argument("--out-dir",  default=OUT_DIR,  help="Output IDX directory (default: set in script)")
    ap.add_argument("--idx",      default=IDX_NAME, help="IDX filename (default: set in script)")
    args = ap.parse_args()

    # Allow overrides via CLI
    BASE_DIR = args.base_dir
    OUT_DIR  = args.out_dir
    IDX_NAME = args.idx
    os.makedirs(OUT_DIR, exist_ok=True)

    y, m, d = map(int, args.date.split("-"))
    write_one_day(y, m, d)

if __name__ == "__main__":
    main()

