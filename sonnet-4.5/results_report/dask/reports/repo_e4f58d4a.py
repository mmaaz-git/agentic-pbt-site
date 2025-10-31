import dask.dataframe.io.parquet.core as core

# Test with empty filters list - the reported issue
parts = []
statistics = []
filters = []

print("Testing apply_filters with empty filters list...")
print(f"Input: parts={parts}, statistics={statistics}, filters={filters}")

try:
    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)
    print(f"Success: filtered_parts={filtered_parts}, filtered_stats={filtered_stats}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()