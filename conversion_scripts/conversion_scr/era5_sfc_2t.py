#!/usr/bin/env python3
import os, re, glob, time
import numpy as np
import xarray as xr
from tqdm import tqdm
import OpenVisus as ov
import pandas as pd

# =========================
# Config
# =========================
BASE_DIR = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc"
OUT_DIR  = "/glade/work/dpanta/era5/idx/2T"
os.makedirs(OUT_DIR, exist_ok=True)

# ERA5 file name pattern:
# e5.oper.an.sfc.128_167_2t.ll025sc.YYYYMMDDHH_YYYYMMDDHH.nc
FILE_REGEX = re.compile(
    r"e5\.oper\.an\.sfc\.128_167_2t\.ll025sc\.(\d{10})_(\d{10})\.nc$"
)

IDX_NAME   = os.path.join(OUT_DIR, "era5_sfc_2T_zip.idx")
ARCO       = "2mb"
FIELD_NAME = "2T"
VAR_NAME   = "VAR_2T"

# Hour-based time range (safe upper bound includes leap years)
TIME_START = 1940 * 365 * 24 + 1
TIME_END   = 2031 * 366 * 24  # generous guard (through ~2030 with leap years)

# Leap-aware origin
BASE0 = np.datetime64("1940-01-01T00:00:00")

def yyyymm_ok(yyyymm: str) -> bool:
    """Keep yyyyMM between 194001 and 202506 inclusive."""
    try:
        val = int(yyyymm)
        return 194001 <= val <= 202506
    except Exception:
        return False

# =========================
# Discover files
# =========================
month_dirs = sorted(
    d for d in glob.glob(os.path.join(BASE_DIR, "*"))
    if os.path.isdir(d) and re.search(r"/(\d{6})$", d) and yyyymm_ok(os.path.basename(d))
)

all_files = []
for mdir in month_dirs:
    files = sorted(glob.glob(os.path.join(mdir, "e5.oper.an.sfc.128_167_2t.ll025sc.*_*.nc")))
    files = [f for f in files if FILE_REGEX.search(os.path.basename(f))]
    all_files.extend(files)

if not all_files:
    raise RuntimeError("No matching NetCDF files found in the requested range.")

print(f"[INFO] Found {len(all_files)} files across {len(month_dirs)} month directories.")

# =========================
# Time indexing (leap-aware, hourly, 1-based)
#   t_abs = 1940*365*24 + days_since_1940*24 + hour + 1
# =========================
def abs_time_index_hours_leapaware(ts_array: np.ndarray) -> np.ndarray:
    ts = pd.to_datetime(ts_array)  # handle calendar and leap years robustly
    # True calendar days since 1940-01-01
    days_since = (
        ts.normalize().values.astype("datetime64[D]") - BASE0.astype("datetime64[D]")
    ) / np.timedelta64(1, "D")
    days_since = days_since.astype(np.int64)
    hours = ts.hour.astype(np.int64)
    base_offset_h = 1940 * 365 * 24
    return base_offset_h + days_since * 24 + hours + 1

# =========================
# Sample first file to get dims/dtype/min/max
# =========================
print(f"[INFO] Sampling first file for dims/dtype/min/max:\n  {all_files[0]}")
with xr.open_dataset(all_files[0]) as ds0:
    if VAR_NAME not in ds0:
        raise RuntimeError(f"Variable '{VAR_NAME}' not found in {all_files[0]}")
    sample = ds0[VAR_NAME].values  # (T, Y, X)
    if sample.ndim != 3:
        raise RuntimeError(f"{VAR_NAME} expected 3D (time,y,x); got shape {sample.shape}")
    T0, Y, X = sample.shape
    dtype = sample.dtype
    vmin0 = np.nanmin(sample); vmax0 = np.nanmax(sample)

fld = ov.Field.fromString(f"""{FIELD_NAME} {str(dtype)} format(row_major) min({vmin0}) max({vmax0})""")

# =========================
# Create IDX once
# =========================
print(f"[INFO] Creating IDX at: {IDX_NAME}")
db_idx = ov.CreateIdx(
    url=IDX_NAME,
    dims=[X, Y],
    fields=[fld],
    compression="raw",     # write raw first; compress at the end
    arco=ARCO,
    time=[TIME_START, TIME_END, "time_%d/"],  # HOUR-BASED window
)

# =========================
# Writer
# =========================
def write_file_to_idx(nc_path: str, db) -> None:
    with xr.open_dataset(nc_path) as ds:
        if VAR_NAME not in ds:
            print(f"[WARN] {VAR_NAME} missing in {nc_path}; skipping.")
            return

        arr = ds[VAR_NAME].values  # (T, Y, X)
        if arr.ndim != 3 or arr.shape[1] != Y or arr.shape[2] != X:
            print(f"[WARN] Skipping {nc_path}; dims mismatch {arr.shape} != (?, {Y}, {X})")
            return

        times = ds.get("time", None)
        if times is None:
            print(f"[WARN] No 'time' coord in {nc_path}; skipping.")
            return

        t_abs = abs_time_index_hours_leapaware(times.values)
        if len(t_abs) != arr.shape[0]:
            raise RuntimeError(f"Time length mismatch in {nc_path}: data T={arr.shape[0]}, time len={len(t_abs)}")

        # Guard the computed indices
        min_t, max_t = int(t_abs.min()), int(t_abs.max())
        if min_t < TIME_START or max_t > TIME_END:
            raise RuntimeError(
                f"Computed hourly indices [{min_t}, {max_t}] outside [{TIME_START}, {TIME_END}] for {nc_path}"
            )

        # Write each hourly slice
        for i in range(arr.shape[0]):
            # Cast if needed to match field dtype (safety)
            slice_i = arr[i, :, :]
            if slice_i.dtype != dtype:
                slice_i = slice_i.astype(dtype, copy=False)
            db.write(slice_i, time=int(t_abs[i]))

# =========================
# Main write loop
# =========================
t0 = time.time()
for f in tqdm(all_files, desc="Writing ERA5 2T to IDX (raw, leap-aware hourly index)"):
    # quick pre-check to avoid opening twice if desired, but we keep a small check
    try:
        write_file_to_idx(f, db_idx)
    except Exception as e:
        print(f"[ERROR] Failed on {f}: {e}")

t1 = time.time()
print(f"[INFO] Raw write complete in {t1 - t0:.2f} s")

# =========================
# Compress with zip (only)
# =========================
print("[INFO] Compressing dataset with 'zip' ...")
t2 = time.time()
db_idx.compressDataset(["zip"])
t3 = time.time()
print(f"[INFO] Compression done in {t3 - t2:.2f} s")
print(f"[DONE] IDX at: {IDX_NAME}")

