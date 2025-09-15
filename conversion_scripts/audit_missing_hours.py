#!/usr/bin/env python3
import os, re, argparse, sys, csv
from datetime import datetime, timedelta
import numpy as np
from glob import glob

# Your hour-index scheme baseline
BASE_YEAR = 1940
BASE0 = np.datetime64(f"{BASE_YEAR:04d}-01-01T00:00:00")

TIME_DIR_RE = re.compile(r"^(\d+)$")

def idx_root_from_path(path: str) -> str:
    """If given a .idx file, return its directory; else return the path itself."""
    if path.endswith(".idx"):
        return os.path.dirname(os.path.abspath(path))
    return os.path.abspath(path)

def parse_present_indices(idx_root: str) -> list[int]:
    """Scan time_* subdirs and extract hour indices as ints."""
    present = []
    for name in os.listdir(idx_root):
        m = TIME_DIR_RE.match(name)
        if m:
            try:
                present.append(int(m.group(1)))
            except ValueError:
                pass
    present.sort()
    return present

def iso_from_hour_index(t: int) -> str:
    """
    Map your absolute 1-based hour index to ISO datetime.
      offset_h = t - (BASE_YEAR*365*24 + 1)
      dt = BASE0 + offset_h hours
    """
    offset = int(t) - (BASE_YEAR * 365 * 24 + 1)
    dt = (BASE0 + np.timedelta64(offset, "h")).astype("M8[ms]")  # millisecond precision for printing
    # Convert to Python datetime for nice formatting
    py_dt = datetime.utcfromtimestamp((dt - np.datetime64('1970-01-01T00:00:00Z')) / np.timedelta64(1, 's'))
    return py_dt.strftime("%Y-%m-%d %H:%M:%S")

def hour_index_from_iso(iso: str) -> int:
    """Inverse: ISO 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM' -> absolute 1-based hour index."""
    # Accept date-only (assume 00:00) or with hour/min/sec
    if len(iso.strip()) <= 10:
        iso = iso.strip() + " 00:00:00"
    dt = np.datetime64(iso.replace(" ", "T"))
    hours_since = int((dt - BASE0) / np.timedelta64(1, "h"))
    return BASE_YEAR * 365 * 24 + hours_since + 1

def contiguous_ranges(sorted_ints: list[int], step: int = 1):
    """Yield (start, end) contiguous ranges over sorted_ints with given step."""
    if not sorted_ints:
        return
    start = prev = sorted_ints[0]
    for x in sorted_ints[1:]:
        if x != prev + step:
            yield (start, prev)
            start = x
        prev = x
    yield (start, prev)

def main():
    ap = argparse.ArgumentParser(
        description="Audit missing hourly timesteps in an OpenVisus IDX time_* tree."
    )
    ap.add_argument("--idx-path", required=True,
                    help="Path to IDX file or its directory (e.g., /.../era5_sfc_2T_zip or /.../era5_sfc_2T_zip.idx)")
    ap.add_argument("--start", default=None,
                    help="Optional start datetime (YYYY-MM-DD or 'YYYY-MM-DD HH:MM'). Default: min present index")
    ap.add_argument("--end", default=None,
                    help="Optional end datetime (YYYY-MM-DD or 'YYYY-MM-DD HH:MM'). Default: max present index")
    ap.add_argument("--csv-out", default="missing_hours.csv",
                    help="CSV filename to write detailed missing hours (default: missing_hours.csv)")
    args = ap.parse_args()

    idx_root = idx_root_from_path(args.idx_path)
    if not os.path.isdir(idx_root):
        print(f"[ERROR] Not a directory: {idx_root}", file=sys.stderr)
        sys.exit(2)

    present = parse_present_indices(idx_root)
    if not present:
        print("[ERROR] No time_* folders found.", file=sys.stderr)
        sys.exit(1)

    # Determine expected range
    if args.start:
        exp_start = hour_index_from_iso(args.start)
    else:
        exp_start = min(present)

    if args.end:
        exp_end = hour_index_from_iso(args.end)
    else:
        exp_end = max(present)

    if exp_end < exp_start:
        exp_start, exp_end = exp_end, exp_start

    present_set = set(present)
    expected = range(exp_start, exp_end + 1)  # inclusive

    missing = [t for t in expected if t not in present_set]

    print(f"[INFO] IDX root: {idx_root}")
    print(f"[INFO] Present indices: {len(present)}")
    print(f"[INFO] Expected range: {exp_start} ({iso_from_hour_index(exp_start)})"
          f"  -> {exp_end} ({iso_from_hour_index(exp_end)})")
    print(f"[INFO] Missing hours: {len(missing)}")

    if not missing:
        print("[OK] No missing hours in the selected range.")
        return

    # Print contiguous gaps
    print("\n[INFO] Contiguous missing ranges:")
    for a, b in contiguous_ranges(missing, step=1):
        print(f"  {a} .. {b}  |  {iso_from_hour_index(a)} .. {iso_from_hour_index(b)}"
              f"  ({b - a + 1} hours)")

    # Write CSV with all missing hours
    with open(args.csv_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["hour_index", "iso_utc"])
        for t in missing:
            writer.writerow([t, iso_from_hour_index(t)])
    print(f"\n[INFO] Wrote detailed list to {args.csv_out}")

if __name__ == "__main__":
    main()

