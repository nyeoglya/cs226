import csv, os
from zoneinfo import ZoneInfo

HOUR_COLS = [f"h{h:02d}" for h in range(8, 19)]  # h08..h18
LOCAL_TZ = ZoneInfo("Europe/London")
UTC_TZ = ZoneInfo("UTC")

DATA_IN = "data/sample_bike_wide.csv"
OUT_DIR = "out"
CSV_OUT = os.path.join(OUT_DIR, "bike_tidy.csv")
PARQUET_OUT = os.path.join(OUT_DIR, "bike_tidy.parquet")

### WRITE_YOUR_CODE
# Import Modules
from datetime import datetime
import pyarrow as pa
import pyarrow.parquet as pq

# Define a lists
csv_result = [['station_id', 'station_name', 'timestamp_local', 'timestamp_utc', 'count']]
station_ids = []
station_names = []
timestamps_local = []
timestamps_utc = []
counts = []

# Read CSV Data
with open(DATA_IN, newline='') as original_file:
    original_file_data = csv.reader(original_file, delimiter=',')

    header = next(original_file_data) # Remove the CSV header

    for rows in original_file_data:
        date = list(map(int, rows[2].split('-')))
        for index, row in enumerate(rows[4:]):
            timestamp_local = datetime(*date, int(HOUR_COLS[index].replace('h', '')), tzinfo=LOCAL_TZ)
            timestamp_utc = timestamp_local.astimezone(UTC_TZ)

            timestamp_local_str = timestamp_local.isoformat()
            timestamp_utc_str = timestamp_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Save datas for Panquet Format
            station_ids.append(rows[0])
            station_names.append(rows[1])
            timestamps_local.append(timestamp_local)
            timestamps_utc.append(timestamp_utc)

            # Check if the row is NA.
            if row not in ['', 'NA', 'N/A']:
                csv_result.append(rows[:2] + [timestamp_local_str, timestamp_utc_str, int(row)])
                counts.append(int(row))
            else:
                csv_result.append(rows[:2] + [timestamp_local_str, timestamp_utc_str, None])
                counts.append(None) 

# Write Data in CSV Format
with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    for re in csv_result:
        writer.writerow(re)

# Write Data in Panquet Format
table = pa.table({
    "station_id": pa.array(station_ids),
    "station_name": pa.array(station_names),
    "timestamp_local": pa.array(timestamps_local, type=pa.timestamp("ns", tz="Europe/London")),
    "timestamp_utc": pa.array(timestamps_utc, type=pa.timestamp("ns", tz="UTC")),
    "count": pa.array(counts, type=pa.int64()),
})
pq.write_table(table, PARQUET_OUT)

# To check the schema(For test purpose)
# loaded = pq.read_table(PARQUET_OUT)
# print(loaded.schema)
# print(loaded)
# print(loaded.to_pandas())
