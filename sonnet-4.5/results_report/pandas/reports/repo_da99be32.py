import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.core.indexers import length_of_indexer

# Test case 1: Empty list with negative stop
target = []
indexer = slice(None, -1, None)

actual_length = len(target[indexer])
computed_length = length_of_indexer(indexer, target)

print(f"Test 1: Empty list with slice(None, -1, None)")
print(f"  target = {target}")
print(f"  len(target[indexer]) = {actual_length}")
print(f"  length_of_indexer(indexer, target) = {computed_length}")
print(f"  ❌ MISMATCH: Expected {actual_length}, got {computed_length}")
print()

# Test case 2: Empty list with more negative stop
target = []
indexer = slice(None, -5, None)

actual_length = len(target[indexer])
computed_length = length_of_indexer(indexer, target)

print(f"Test 2: Empty list with slice(None, -5, None)")
print(f"  target = {target}")
print(f"  len(target[indexer]) = {actual_length}")
print(f"  length_of_indexer(indexer, target) = {computed_length}")
print(f"  ❌ MISMATCH: Expected {actual_length}, got {computed_length}")
print()

# Test case 3: Small list with large negative stop
target = [1, 2]
indexer = slice(None, -5, None)

actual_length = len(target[indexer])
computed_length = length_of_indexer(indexer, target)

print(f"Test 3: List [1,2] with slice(None, -5, None)")
print(f"  target = {target}")
print(f"  len(target[indexer]) = {actual_length}")
print(f"  length_of_indexer(indexer, target) = {computed_length}")
print(f"  ❌ MISMATCH: Expected {actual_length}, got {computed_length}")
print()

# Test case 4: Working case for comparison
target = [1, 2, 3]
indexer = slice(None, -1, None)

actual_length = len(target[indexer])
computed_length = length_of_indexer(indexer, target)

print(f"Test 4: List [1,2,3] with slice(None, -1, None)")
print(f"  target = {target}")
print(f"  len(target[indexer]) = {actual_length}")
print(f"  length_of_indexer(indexer, target) = {computed_length}")
print(f"  ✓ CORRECT: Both are {actual_length}")