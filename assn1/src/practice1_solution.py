#!/usr/bin/env python3
# Lab 01 — Reference Solution (no pandas)

import csv
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, Iterator, List, Optional, Dict
from zoneinfo import ZoneInfo

import pyarrow as pa
import pyarrow.parquet as pq

# ---------- Constants ----------
HOUR_COLS = [f"h{h:02d}" for h in range(8, 19)]
LOCAL_TZ = ZoneInfo("Europe/London")
UTC_TZ = ZoneInfo("UTC")

DATA_IN = "data/sample_bike_wide.csv"
OUT_DIR = "out"
CSV_OUT = os.path.join(OUT_DIR, "bike_tidy.csv")
PARQUET_OUT = os.path.join(OUT_DIR, "bike_tidy.parquet")

MISSING = {"", "NA", "N/A"}


# ---------- Tidy row model ----------
@dataclass
class TidyRow:
    station_id: str
    station_name: str
    timestamp_local: datetime  # tz-aware Europe/London
    timestamp_utc: datetime    # tz-aware UTC
    count: Optional[int]       # may be None


# ---------- Core helpers ----------
def parse_count(x: str) -> Optional[int]:
    x = (x or "").strip()
    return None if x in MISSING else int(x)

def hour_to_dt(date_str: str, hour: int) -> datetime:
    y, m, d = map(int, date_str.split("-"))
    return datetime(y, m, d, hour, 0, 0, tzinfo=LOCAL_TZ)

def wide_csv_rows(path: str) -> Iterator[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            yield row

def rows_to_tidy(wide_rows: Iterable[Dict[str, str]]) -> Iterator[TidyRow]:
    for row in wide_rows:
        sid = row["station_id"]          # keep leading zeros
        sname = row["station_name"]
        date_str = row["date"]           # YYYY-MM-DD
        for col in HOUR_COLS:
            hour = int(col[1:])          # e.g., "h08" -> 8
            local_dt = hour_to_dt(date_str, hour)
            utc_dt = local_dt.astimezone(UTC_TZ)
            cnt = parse_count(row.get(col, ""))
            yield TidyRow(sid, sname, local_dt, utc_dt, cnt)


# ---------- Writers ----------
def write_csv(tidy: Iterable[TidyRow], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["station_id", "station_name", "timestamp_local", "timestamp_utc", "count"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in tidy:
            # local ISO8601 with offset; UTC ends with 'Z'
            local_iso = t.timestamp_local.replace(microsecond=0).isoformat()
            utc_iso = t.timestamp_utc.astimezone(UTC_TZ).strftime("%Y-%m-%dT%H:%M:%SZ")
            w.writerow({
                "station_id": t.station_id,
                "station_name": t.station_name,
                "timestamp_local": local_iso,
                "timestamp_utc": utc_iso,
                "count": "" if t.count is None else t.count,
            })

def write_parquet(tidy_list: List[TidyRow], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    table = pa.table({
        "station_id": pa.array([t.station_id for t in tidy_list], pa.string()),
        "station_name": pa.array([t.station_name for t in tidy_list], pa.string()),
        # Store proper timezone-aware timestamps
        "timestamp_local": pa.array([t.timestamp_local for t in tidy_list],
                                    type=pa.timestamp("ns", tz="Europe/London")),
        "timestamp_utc": pa.array([t.timestamp_utc for t in tidy_list],
                                  type=pa.timestamp("ns", tz="UTC")),
        # Nullable int32 for missing counts
        "count": pa.array([t.count for t in tidy_list], type=pa.int32()),
    })
    pq.write_table(table, path)


# ---------- Main ----------
def main() -> None:
    wide = list(wide_csv_rows(DATA_IN))
    tidy_iter = rows_to_tidy(wide)

    # Write CSV (streaming)
    # Also collect rows to list once for Parquet (small dataset)
    collected: List[TidyRow] = []
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["station_id", "station_name", "timestamp_local", "timestamp_utc", "count"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for t in tidy_iter:
            collected.append(t)
            local_iso = t.timestamp_local.replace(microsecond=0).isoformat()
            utc_iso = t.timestamp_utc.astimezone(UTC_TZ).strftime("%Y-%m-%dT%H:%M:%SZ")
            w.writerow({
                "station_id": t.station_id,
                "station_name": t.station_name,
                "timestamp_local": local_iso,
                "timestamp_utc": utc_iso,
                "count": "" if t.count is None else t.count,
            })

    write_parquet(collected, PARQUET_OUT)


if __name__ == "__main__":
    main()
