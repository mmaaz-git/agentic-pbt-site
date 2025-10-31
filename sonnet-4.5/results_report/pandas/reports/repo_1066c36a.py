import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from pandas.core.indexers import length_of_indexer

# Test the specific failing case from the bug report
target = np.array([0])
indexer = slice(1, 0, None)

computed = length_of_indexer(indexer, target)
actual = len(target[indexer])

print(f"Test case: slice(1, 0, None) on array [0]")
print(f"Computed length: {computed}")
print(f"Actual length: {actual}")
print(f"Bug confirmed: {computed} != {actual}")
print()

# Test additional cases to understand the scope
test_cases = [
    (np.arange(5), slice(None, None, -1), "Reverse slice on [0,1,2,3,4]"),
    (np.arange(3), slice(10, 20, None), "Out of bounds slice on [0,1,2]"),
    (np.arange(4), slice(-1, 0, None), "Negative to positive empty slice on [0,1,2,3]"),
    (np.array([42]), slice(None, None, -1), "Reverse slice on single element [42]"),
    (np.arange(10), slice(5, 3, None), "Empty slice(5, 3, None) on [0..9]"),
]

print("Additional test cases:")
print("-" * 50)
for target, indexer, description in test_cases:
    computed = length_of_indexer(indexer, target)
    actual = len(target[indexer])
    match = "✓" if computed == actual else "✗"
    print(f"{description}:")
    print(f"  Indexer: {indexer}")
    print(f"  Computed: {computed}, Actual: {actual} {match}")
    if computed != actual:
        print(f"  ERROR: Expected {actual}, got {computed}")
    print()