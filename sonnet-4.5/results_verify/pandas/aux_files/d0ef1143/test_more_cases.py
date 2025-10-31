import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from pandas.core.indexers import length_of_indexer

print("Testing various slice cases:\n")

test_cases = [
    (np.array([0, 1, 2, 3, 4]), slice(1, 0, None), "Empty slice (start > stop)"),
    (np.array([0, 1, 2, 3, 4]), slice(2, 2, None), "Empty slice (start == stop)"),
    (np.array([0, 1, 2, 3, 4]), slice(5, 10, None), "Start beyond array"),
    (np.array([0, 1, 2, 3, 4]), slice(-1, 0, None), "Negative to positive, empty"),
    (np.array([0, 1, 2, 3, 4]), slice(None, None, -1), "Reverse entire array"),
    (np.array([0, 1, 2, 3, 4]), slice(3, 1, -1), "Reverse slice"),
    (np.array([0]), slice(None, None, -1), "Reverse single element"),
    (np.array([]), slice(None, None, None), "Empty array"),
    (np.array([0, 1, 2]), slice(10, 20, None), "Both indices out of bounds"),
]

for target, indexer, description in test_cases:
    try:
        computed = length_of_indexer(indexer, target)
        actual = len(target[indexer])
        match = "✓" if computed == actual else "✗"
        print(f"{match} {description}")
        print(f"   Target length: {len(target)}, Indexer: {indexer}")
        print(f"   Computed: {computed}, Actual: {actual}")
        if computed != actual:
            print(f"   ERROR: Expected {actual} but got {computed}")
    except Exception as e:
        print(f"✗ {description}")
        print(f"   Error: {e}")
    print()