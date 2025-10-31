import dask.dataframe.io.parquet.core as core

# Test with None filters
parts = ["part1", "part2"]
statistics = [{"col1": {"min": 1, "max": 10}}, {"col1": {"min": 11, "max": 20}}]
filters = None

print("Testing with filters=None...")
try:
    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)
    print(f"Success! Got: parts={filtered_parts}, stats={filtered_stats}")
except (IndexError, TypeError, AttributeError) as e:
    print(f"Error raised: {type(e).__name__}: {e}")

# Test with empty list
filters = []
print("\nTesting with filters=[]...")
try:
    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)
    print(f"Success! Got: parts={filtered_parts}, stats={filtered_stats}")
except (IndexError, TypeError, AttributeError) as e:
    print(f"Error raised: {type(e).__name__}: {e}")

# Test with valid filters
filters = [("col1", ">", 5)]
print("\nTesting with filters=[('col1', '>', 5)]...")
try:
    filtered_parts, filtered_stats = core.apply_filters(parts, statistics, filters)
    print(f"Success! Got: parts={filtered_parts}, stats={filtered_stats}")
except (IndexError, TypeError, AttributeError) as e:
    print(f"Error raised: {type(e).__name__}: {e}")