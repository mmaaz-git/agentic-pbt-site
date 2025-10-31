from pandas.arrays import SparseArray
import numpy as np

arr1 = SparseArray([0, 0, 1], fill_value=0)
arr2 = SparseArray([2, 2, 3], fill_value=2)

print("Before concatenation:")
print(f"arr1: {list(arr1.to_dense())}")
print(f"arr1.fill_value: {arr1.fill_value}")
print(f"arr2: {list(arr2.to_dense())}")
print(f"arr2.fill_value: {arr2.fill_value}")

result = SparseArray._concat_same_type([arr1, arr2])

print("\nAfter concatenation:")
print(f"Result: {list(result.to_dense())}")
print(f"Result.fill_value: {result.fill_value}")

print("\nExpected result: [0, 0, 1, 2, 2, 3]")
print("Actual result:   {}".format(list(result.to_dense())))

expected = np.concatenate([arr1.to_dense(), arr2.to_dense()])
print(f"\nDoes result match expected? {np.array_equal(result.to_dense(), expected)}")