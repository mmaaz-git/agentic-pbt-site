import numpy as np
from pandas.core.util.hashing import combine_hash_arrays

# Test 1: Empty iterator with num_items=1 (should raise AssertionError but doesn't)
print("Test 1: Empty iterator with num_items=1")
result = combine_hash_arrays(iter([]), 1)
print(f"Result: {result}")
print(f"Result type: {type(result)}")
print(f"Result shape: {result.shape}")
print(f"Expected: AssertionError with message 'Fed in wrong num_items'")
print(f"Actual: Returns empty array without error")
print()

# Test 2: Empty iterator with various num_items values
print("Test 2: Empty iterator with various num_items values")
for num_items in [2, 5, 10]:
    result = combine_hash_arrays(iter([]), num_items)
    print(f"num_items={num_items}: result={result}, shape={result.shape}")
print()

# Test 3: Empty iterator with num_items=0 (correct case)
print("Test 3: Empty iterator with num_items=0 (should succeed)")
result = combine_hash_arrays(iter([]), 0)
print(f"Result: {result}")
print(f"This correctly returns an empty array")
print()

# Test 4: Non-empty case with wrong num_items (for comparison)
print("Test 4: Non-empty iterator with wrong num_items (should raise AssertionError)")
arr = np.array([1, 2, 3], dtype=np.uint64)
try:
    result = combine_hash_arrays(iter([arr]), 2)
    print(f"Unexpectedly succeeded: {result}")
except AssertionError as e:
    print(f"Correctly raised AssertionError: {e}")
print()

# Test 5: Non-empty case with correct num_items (should succeed)
print("Test 5: Non-empty iterator with correct num_items (should succeed)")
arr = np.array([1, 2, 3], dtype=np.uint64)
result = combine_hash_arrays(iter([arr]), 1)
print(f"Result: {result}")
print(f"This correctly processes the array")