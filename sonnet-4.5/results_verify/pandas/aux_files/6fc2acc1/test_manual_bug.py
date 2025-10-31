from pandas.api.indexers import FixedForwardWindowIndexer

indexer = FixedForwardWindowIndexer(window_size=-1)
start, end = indexer.get_window_bounds(num_values=2)

print(f"start: {start}")
print(f"end: {end}")
print(f"\nInvariant violated: start[1] = {start[1]} > end[1] = {end[1]}")

# Additional test with different negative values
print("\nAdditional tests:")
for window_size in [-2, -5, -10]:
    indexer = FixedForwardWindowIndexer(window_size=window_size)
    start, end = indexer.get_window_bounds(num_values=5)
    print(f"window_size={window_size}:")
    print(f"  start: {start}")
    print(f"  end: {end}")
    violations = []
    for i in range(len(start)):
        if start[i] > end[i]:
            violations.append(f"index {i}: start[{i}]={start[i]} > end[{i}]={end[i]}")
    if violations:
        print(f"  Violations: {', '.join(violations)}")