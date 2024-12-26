import duckdb

duckdb.read_parquet('./Segment_Files/*/*parquet')
print(duckdb)
