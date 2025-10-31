import numpy as np
from pandas.api.indexers import FixedForwardWindowIndexer

# Test different negative window sizes
for window_size in [-1, -2, -5, -10]:
    print(f"\n=== Testing window_size={window_size} ===")
    for num_values in [1, 2, 3, 5, 10]:
        indexer = FixedForwardWindowIndexer(window_size=window_size)
        start, end = indexer.get_window_bounds(num_values)

        # Check for violations
        violations = []
        for i in range(len(start)):
            if start[i] > end[i]:
                violations.append((i, start[i], end[i]))

        if violations:
            print(f"  num_values={num_values}: VIOLATED - {violations}")
        else:
            print(f"  num_values={num_values}: OK - start={start}, end={end}")

# Test zero window size
print(f"\n=== Testing window_size=0 ===")
for num_values in [1, 2, 3, 5, 10]:
    indexer = FixedForwardWindowIndexer(window_size=0)
    start, end = indexer.get_window_bounds(num_values)
    print(f"  num_values={num_values}: start={start}, end={end}")

# Test positive window sizes for comparison
print(f"\n=== Testing window_size=2 (positive) ===")
for num_values in [1, 2, 3, 5]:
    indexer = FixedForwardWindowIndexer(window_size=2)
    start, end = indexer.get_window_bounds(num_values)
    print(f"  num_values={num_values}: start={start}, end={end}")