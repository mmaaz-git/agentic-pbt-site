from dask.dataframe.io.parquet.core import apply_filters

# Test case 1: Empty everything (the minimal failing case)
parts = []
statistics = []
filters = []

print("Test case 1: Empty parts, statistics, and filters")
print(f"parts = {parts}")
print(f"statistics = {statistics}")
print(f"filters = {filters}")
print()

try:
    result_parts, result_stats = apply_filters(parts, statistics, filters)
    print("Result:")
    print(f"  result_parts = {result_parts}")
    print(f"  result_stats = {result_stats}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60 + "\n")

# Test case 2: Non-empty parts with empty filters
parts = [{"piece": 1}, {"piece": 2}]
statistics = [{"columns": []}, {"columns": []}]
filters = []

print("Test case 2: Non-empty parts with empty filters")
print(f"parts = {parts}")
print(f"statistics = {statistics}")
print(f"filters = {filters}")
print()

try:
    result_parts, result_stats = apply_filters(parts, statistics, filters)
    print("Result:")
    print(f"  result_parts = {result_parts}")
    print(f"  result_stats = {result_stats}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()