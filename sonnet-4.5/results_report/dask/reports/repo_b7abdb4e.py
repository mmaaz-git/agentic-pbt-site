import dask.dataframe.io.parquet.core as parquet_core

parts = ['part1']
statistics = [{'columns': []}]
filters = []

print("Testing apply_filters with empty filters list:")
print(f"parts = {parts}")
print(f"statistics = {statistics}")
print(f"filters = {filters}")
print()

try:
    out_parts, out_statistics = parquet_core.apply_filters(parts, statistics, filters)
    print(f"Result: out_parts = {out_parts}, out_statistics = {out_statistics}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()