import numpy as np
from pandas.core.util.hashing import combine_hash_arrays

# Test empty iterator with num_items=1
result = combine_hash_arrays(iter([]), 1)
print(f"Result: {result}")
print(f"Expected: AssertionError with message 'Fed in wrong num_items'")
print(f"Actual: Returns empty array without error")

# Compare with non-empty case
arr = np.array([1, 2, 3], dtype=np.uint64)
try:
    result = combine_hash_arrays(iter([arr]), 2)
except AssertionError as e:
    print(f"\nFor comparison, non-empty case correctly raises: {e}")

# Test the correct empty case
result_correct = combine_hash_arrays(iter([]), 0)
print(f"\nCorrect empty case (num_items=0): {result_correct}")

# Test with multiple positive values for num_items and empty iterator
print("\nTesting multiple positive num_items with empty iterator:")
for num in [1, 2, 5, 10]:
    try:
        result = combine_hash_arrays(iter([]), num)
        print(f"  num_items={num}: Returned {result} (should have raised AssertionError)")
    except AssertionError as e:
        print(f"  num_items={num}: Correctly raised AssertionError: {e}")