import numpy as np

print("Testing numpy's behavior with where parameter")
print("=" * 60)

arr = np.array([1, 2, 3], dtype=np.int64)
mask = np.ones(3, dtype=bool)  # All True = all masked

print(f"Array: {arr}")
print(f"Mask: {mask} (all True)")
print(f"~mask: {~mask} (inverted mask for where parameter)")
print()

# What numpy does with where parameter
result_sum = np.sum(arr, where=~mask)
print(f"np.sum(arr, where=~mask) = {result_sum}")
print(f"Type: {type(result_sum)}")
print()

result_prod = np.prod(arr, where=~mask)
print(f"np.prod(arr, where=~mask) = {result_prod}")
print(f"Type: {type(result_prod)}")
print()

# Show what happens with some unmasked values
mask2 = np.array([True, False, True], dtype=bool)
print(f"\nWith partial mask: {mask2}")
print(f"~mask2: {~mask2}")
result_sum2 = np.sum(arr, where=~mask2)
print(f"np.sum(arr, where=~mask2) = {result_sum2} (only sums arr[1]=2)")

result_prod2 = np.prod(arr, where=~mask2)
print(f"np.prod(arr, where=~mask2) = {result_prod2} (only multiplies arr[1]=2)")