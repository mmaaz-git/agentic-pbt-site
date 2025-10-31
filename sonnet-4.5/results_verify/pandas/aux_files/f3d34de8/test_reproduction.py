import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from pandas.core.indexers import length_of_indexer

# Test the specific case from the bug report
target = np.array([0])
indexer = slice(1, 0, None)

computed = length_of_indexer(indexer, target)
actual = len(target[indexer])

print(f"Test case from bug report:")
print(f"Target: {target}")
print(f"Indexer: {indexer}")
print(f"Computed: {computed}")
print(f"Actual: {actual}")
print(f"Bug: {computed} != {actual}")
print()

# Test the case found by our hypothesis test
target2 = np.array([0])
indexer2 = slice(None, None, -1)

computed2 = length_of_indexer(indexer2, target2)
actual2 = len(target2[indexer2])

print(f"Test case found by Hypothesis:")
print(f"Target: {target2}")
print(f"Indexer: {indexer2}")
print(f"Computed: {computed2}")
print(f"Actual: {actual2}")
print(f"Bug: {computed2} != {actual2}")