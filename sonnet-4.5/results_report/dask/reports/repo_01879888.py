from dask.dataframe.io.parquet.core import sorted_columns

# Test case from the bug report - min is valid but max is None
statistics = [{'columns': [{'name': '0', 'min': 0, 'max': None}]}]

try:
    result = sorted_columns(statistics)
    print(f"Result: {result}")
except TypeError as e:
    print(f"TypeError: {e}")
    import traceback
    traceback.print_exc()