#!/usr/bin/env python3
import argparse, re, sys, os
from datetime import datetime, timedelta, date
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import xarray as xr

# ---------- parse "missing ranges" ----------
RANGE_RE = re.compile(
    r"^\s*(\d+)\s*\.\.\s*(\d+)\s*\|\s*"
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\.\.\s*"
    r"(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s*\(\d+\s*hours\)\s*$"
)

def parse_ranges(path):
    out = []
    with open(path, "r") as f:
        for line in f:
            m = RANGE_RE.match(line)
            if not m:
                continue
            s = datetime.strptime(m.group(3), "%Y-%m-%d %H:%M:%S")
            e = datetime.strptime(m.group(4), "%Y-%m-%d %H:%M:%S")
            if e < s: s, e = e, s
            out.append((s, e))
    if not out:
        print(f"[ERROR] no ranges parsed from {path}", file=sys.stderr)
        sys.exit(2)
    return out

def days_from_ranges(ranges):
    days = set()
    for s, e in ranges:
        d = s.date()
        while d <= e.date():
            days.add(d)
            d = d + timedelta(days=1)
    return sorted(days)

def group_days_by_month(days):
    buckets = defaultdict(list)
    for d in days:
        buckets[(d.year, d.month)].append(d)
    for k in buckets:
        buckets[k] = sorted(buckets[k])
    return dict(buckets)

# ---------- core fast writer ----------
BASE_YEAR = 1940

def abs_time_index_hours_leapaware(ts_array):
    ts = ts_array.astype("datetime64[ns]")
    base = np.datetime64(f"{BASE_YEAR:04d}-01-01T00:00:00").astype("datetime64[ns]")
    hours_since = ((ts - base) / np.timedelta64(1, "h")).astype(np.int64)
    return hours_since + (BASE_YEAR * 365 * 24 + 1)

def month_nc_path(base_dir, year, month):
    yyyymm = f"{year:04d}{month:02d}"
    month_dir = os.path.join(base_dir, yyyymm)
    if not os.path.isdir(month_dir):
        raise FileNotFoundError(f"Month directory not found: {month_dir}")
    import glob, os, re
    FILE_REGEX = re.compile(r"e5\.oper\.an\.sfc\.128_167_2t\.ll025sc\.(\d{10})_(\d{10})\.nc$")
    cands = sorted(
        f for f in glob.glob(os.path.join(month_dir, "e5.oper.an.sfc.128_167_2t.ll025sc.*_*.nc"))
        if FILE_REGEX.search(os.path.basename(f))
    )
    if not cands:
        raise FileNotFoundError(f"No ERA5 2T file in {month_dir}")
    return cands[0]

def ensure_idx(idx_path, out_dir, sample_nc, var_name="VAR_2T", field_name="2T", arco="2mb"):
    import OpenVisus as ov
    os.makedirs(out_dir, exist_ok=True)
    if os.path.exists(idx_path):
        return ov.LoadDataset(idx_path)
    # create from sample
    with xr.open_dataset(sample_nc) as ds:
        if var_name not in ds:
            raise RuntimeError(f"Variable '{var_name}' not in {sample_nc}")
        a = ds[var_name].values  # (T,Y,X)
        _, Y, X = a.shape
        dtype = a.dtype
        vmin = np.nanmin(a)
        vmax = np.nanmax(a)
    fld = ov.Field.fromString(f"{field_name} {str(dtype)} format(row_major) min({vmin}) max({vmax})")
    TIME_START = BASE_YEAR * 365 * 24 + 1
    TIME_END   = 2031 * 366 * 24
    return ov.CreateIdx(
        url=idx_path, dims=[X, Y], fields=[fld],
        compression="raw", arco=arco, time=[TIME_START, TIME_END, "time_%d/"]
    )

def write_month(args):
    """
    Worker run in a separate process.
    args = (year, month, days, base_dir, out_dir, idx_path, var_name, field_name, compress_mode)
    """
    (year, month, days, base_dir, out_dir, idx_path, var_name, field_name, compress_mode) = args
    import OpenVisus as ov

    nc = month_nc_path(base_dir, year, month)
    ds = xr.open_dataset(nc)  # decode_times=True by default
    try:
        if var_name not in ds:
            raise RuntimeError(f"{var_name} not in {nc}")
        times_all = ds["time"].values  # monthly datetime64
        data_all  = ds[var_name]       # (T,Y,X)

        # build boolean mask for all hours we need in this month
        # we include full days [00:00 .. 23:00]
        m = np.zeros(times_all.shape[0], dtype=bool)
        day_set = set(days)
        # vectorized trick: compute dates for all timestamps, then mark if in set
        times_dates = times_all.astype("datetime64[D]").astype(object)  # list of date objects
        for i, d in enumerate(times_dates):
            if d in day_set:
                m[i] = True

        if not m.any():
            return f"[{year}-{month:02d}] nothing to write"

        # load only needed hours into memory (contiguous gather is fine: one month ~<= 744)
        arr = data_all.values[m, :, :]           # (H,Y,X)
        ts  = times_all[m]                       # (H,)

        # init/open IDX once
        db = ensure_idx(idx_path, out_dir, nc, var_name=var_name, field_name=field_name)

        # compute absolute time indices
        t_abs = abs_time_index_hours_leapaware(ts)

        # write loop (no per-timestep compression by default)
        # We keep array order as (Y,X) -> IDX dims were [X, Y], and OpenVisus expects row_major.
        # If your output looks flipped, uncomment the transpose below.
        # arr = arr.transpose(0,2,1)  # (H,X,Y)

        for i in range(arr.shape[0]):
            db.write(arr[i, :, :], time=int(t_abs[i]))
            if compress_mode == "per-timestep":
                db.compressDataset(["zip"], timestep=int(t_abs[i]))

        if compress_mode == "end":
            # Fall back to per-timestep compress in a tight loop (still cheaper than opening files repeatedly)
            for t in t_abs.tolist():
                db.compressDataset(["zip"], timestep=int(t))

        return f"[{year}-{month:02d}] wrote {arr.shape[0]} hours for {len(days)} day(s)"
    finally:
        ds.close()

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(
        description="Fast re-writer for missing ERA5 2T days (groups by month, opens each file once)."
    )
    ap.add_argument("--ranges", required=True, help="Text file with 'start..end | ISO..ISO (N hours)' lines.")
    ap.add_argument("--base-dir", required=True, help="ERA5 base dir (yyyyMM subdirs).")
    ap.add_argument("--out-dir",  required=True, help="Output IDX directory.")
    ap.add_argument("--idx",      required=True, help="IDX filename (e.g., /.../era5_sfc_2T_zip.idx)")
    ap.add_argument("--var",      default="VAR_2T", help="NetCDF variable name (default: VAR_2T)")
    ap.add_argument("--field",    default="2T",     help="IDX field name (default: 2T)")
    ap.add_argument("--jobs",     type=int, default=4, help="Parallel processes (default: 4)")
    ap.add_argument("--compress", choices=["none","per-timestep","end"], default="none",
                    help="Compression strategy (default: none)")
    args = ap.parse_args()

    ranges = parse_ranges(args.ranges)
    days   = days_from_ranges(ranges)
    groups = group_days_by_month(days)

    print(f"[INFO] unique days: {len(days)} across {len(groups)} month(s)")
    for (y,m), ds_ in sorted(groups.items()):
        print(f"  - {y}-{m:02d}: {len(ds_)} day(s)")

    # parallel by month (good balance of IO/CPU)
    work = [ (y, m, groups[(y,m)], args.base_dir, args.out_dir, args.idx, args.var, args.field, args.compress)
             for (y,m) in groups.keys() ]

    if args.jobs <= 1:
        for w in work:
            print(write_month(w))
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            futs = [ex.submit(write_month, w) for w in work]
            for fut in as_completed(futs):
                print(fut.result())

if __name__ == "__main__":
    main()

