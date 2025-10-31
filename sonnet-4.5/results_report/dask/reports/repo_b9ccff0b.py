from dask.dataframe.io.parquet.core import sorted_columns

# Test case 1: None in both min and max for first row group, valid min but None max in second
print("Test case 1:")
print("-" * 50)
statistics = [
    {'columns': [{'name': 'test_col', 'min': None, 'max': None}]},
    {'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}
]

try:
    result = sorted_columns(statistics, columns=['test_col'])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\n")

# Test case 2: Single row group with valid min but None max
print("Test case 2:")
print("-" * 50)
statistics = [
    {'columns': [{'name': 'test_col', 'min': 0, 'max': None}]}
]

try:
    result = sorted_columns(statistics, columns=['test_col'])
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")