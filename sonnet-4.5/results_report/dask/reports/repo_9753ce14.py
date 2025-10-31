from dask.dataframe.io.parquet.core import sorted_columns

# Minimal reproduction of the bug
statistics = [{"columns": [{"name": "a", "min": 0, "max": None}]}]

print("Input statistics:")
print(f"  statistics = {statistics}")
print()
print("Calling sorted_columns(statistics)...")
print()

try:
    result = sorted_columns(statistics)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")
    import traceback
    print()
    print("Full traceback:")
    traceback.print_exc()