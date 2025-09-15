#!/usr/bin/env python3
import argparse, re, sys, subprocess, os
from datetime import datetime, timedelta
from collections import defaultdict

# ---- hour-index <-> datetime mapping (matches the audit script) ----
BASE_YEAR = 1940
BASE0 = datetime(BASE_YEAR, 1, 1, 0, 0, 0)  # UTC

def dt_from_hour_index(t: int) -> datetime:
    offset_h = t - (BASE_YEAR * 365 * 24 + 1)
    return BASE0 + timedelta(hours=offset_h)

# ---- parse the audit text lines you pasted ----
RANGE_RE = re.compile(
    r"^\s*(\d+)\s*\.\.\s*(\d+)\s*\|\s*([0-9:\-\s]+)\s*\.\.\s*([0-9:\-\s]+)\s*\(\d+\s*hours\)\s*$"
)

def parse_ranges(path):
    ranges = []
    with open(path, "r") as f:
        for line in f:
            m = RANGE_RE.match(line)
            if not m:
                continue
            start_idx = int(m.group(1))
            end_idx   = int(m.group(2))
            # we trust the indices; dates are just for human check
            ranges.append((start_idx, end_idx))
    if not ranges:
        print(f"[ERROR] No ranges parsed from: {path}", file=sys.stderr)
        sys.exit(2)
    return ranges

def expand_hours(ranges):
    """Return generator of all missing hour indices (inclusive ranges)."""
    for a, b in ranges:
        if b < a:
            a, b = b, a
        for t in range(a, b + 1):
            yield t

def unique_days_from_hours(ranges):
    days = set()
    for t in expand_hours(ranges):
        d = dt_from_hour_index(t).date()  # YYYY-MM-DD
        days.add(d.isoformat())
    return sorted(days)

def write_missing_lists(ranges, hours_csv="missing_hours.csv", days_txt="missing_days.txt"):
    # write hours CSV
    with open(hours_csv, "w") as f:
        f.write("hour_index,iso_utc\n")
        for t in expand_hours(ranges):
            f.write(f"{t},{dt_from_hour_index(t).strftime('%Y-%m-%d %H:%M:%S')}\n")
    # write unique days
    days = unique_days_from_hours(ranges)
    with open(days_txt, "w") as f:
        for d in days:
            f.write(d + "\n")
    return hours_csv, days_txt, days

def run_for_days(days, cmd_template, jobs=1, dry_run=True, stop_on_error=False, env=None):
    """
    cmd_template example:
      'python convert_one_day.py --date {date}'
    where {date} will be replaced by 'YYYY-MM-DD'

    jobs>1 uses a simple process pool (no external deps).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def one(day):
        cmd = cmd_template.format(date=day)
        if dry_run:
            return (day, 0, f"[DRY-RUN] {cmd}")
        try:
            res = subprocess.run(cmd, shell=True, check=True, env=env)
            return (day, res.returncode, "")
        except subprocess.CalledProcessError as e:
            return (day, e.returncode, str(e))

    results = []
    if jobs <= 1:
        for d in days:
            r = one(d)
            print(f"[{d}] rc={r[1]} {r[2]}")
            results.append(r)
            if not dry_run and stop_on_error and r[1] != 0:
                print("[STOP] Aborting due to error.")
                break
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futs = {ex.submit(one, d): d for d in days}
            for fut in as_completed(futs):
                d = futs[fut]
                r = fut.result()
                print(f"[{d}] rc={r[1]} {r[2]}")
                results.append(r)
                if not dry_run and stop_on_error and r[1] != 0:
                    # Best-effort: we can't cancel all futures cleanly here.
                    print("[STOP] Errors occurred; further tasks may still be running.")
                    break
    return results

def main():
    ap = argparse.ArgumentParser(description="Rebuild missing IDX timesteps from audit ranges.")
    ap.add_argument("--ranges", required=True,
                    help="Path to text file that contains the audit 'start..end | ... (N hours)' lines.")
    ap.add_argument("--hours-csv", default="missing_hours.csv",
                    help="Where to write full list of missing hours.")
    ap.add_argument("--days-txt", default="missing_days.txt",
                    help="Where to write unique missing days.")
    ap.add_argument("--execute", action="store_true",
                    help="If set, actually run the per-day conversion command.")
    ap.add_argument("--cmd", default=None,
                    help="Command template to run per day, e.g. 'python convert_one_day.py --date {date}'. Must contain {date}.")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Parallel workers for per-day conversion (default: 1).")
    ap.add_argument("--stop-on-error", action="store_true",
                    help="Stop submitting new work when a command fails.")
    args = ap.parse_args()

    ranges = parse_ranges(args.ranges)
    hours_csv, days_txt, days = write_missing_lists(ranges, args.hours_csv, args.days_txt)
    print(f"[INFO] Wrote hours to: {hours_csv}")
    print(f"[INFO] Wrote days  to: {days_txt}")
    print(f"[INFO] Unique days missing: {len(days)} (first: {days[0]}, last: {days[-1]})")

    if args.execute:
        if not args.cmd or "{date}" not in args.cmd:
            print("[ERROR] --execute requires a --cmd with a {date} placeholder.", file=sys.stderr)
            sys.exit(3)
        print(f"[INFO] Executing per-day converter with {args.jobs} job(s).")
        run_for_days(days, args.cmd, jobs=args.jobs, dry_run=False, stop_on_error=args.stop_on_error)

if __name__ == "__main__":
    main()

