import dask.dataframe.io.parquet.core as core

parts = []
statistics = []
filters = []

print("Testing with empty parts, statistics, and filters...")
try:
    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)
    print(f"Success! Got: parts={filtered_parts}, stats={filtered_stats}")
except IndexError as e:
    print(f"IndexError raised: {e}")

# Also test with non-empty parts and statistics but empty filters
print("\nTesting with non-empty parts and statistics but empty filters...")
parts = ["part1", "part2"]
statistics = [{"col1": {"min": 1, "max": 10}}, {"col1": {"min": 11, "max": 20}}]
filters = []

try:
    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)
    print(f"Success! Got: parts={filtered_parts}, stats={filtered_stats}")
except IndexError as e:
    print(f"IndexError raised: {e}")