from pandas.arrays import SparseArray
import numpy as np

# Test case demonstrating the bug
arr1 = SparseArray([0, 0, 1], fill_value=0)
arr2 = SparseArray([2, 2, 3], fill_value=2)

result = SparseArray._concat_same_type([arr1, arr2])

print("Input arrays:")
print(f"  arr1: {list(arr1.to_dense())} (fill_value={arr1.fill_value})")
print(f"  arr2: {list(arr2.to_dense())} (fill_value={arr2.fill_value})")
print()
print("Concatenation result:")
print(f"  Result: {list(result.to_dense())} (fill_value={result.fill_value})")
print()
print("Expected result:")
print(f"  Expected: [0, 0, 1, 2, 2, 3]")
print()
print("Analysis:")
expected = np.concatenate([arr1.to_dense(), arr2.to_dense()])
actual = result.to_dense()
if not np.array_equal(expected, actual):
    print(f"  ERROR: Data loss detected!")
    print(f"  Missing values: {[e for e, a in zip(expected, actual) if e != a]}")
else:
    print("  OK: Data preserved correctly")