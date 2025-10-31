import numpy as np
from pandas.core.array_algos.masked_accumulations import cumsum

# Create test data
values = np.array([1, 2, 3, 4, 5])
mask = np.array([False, True, False, True, False])

print("Before cumsum:")
print(f"  values: {values}")
print(f"  mask:   {mask}")

# Call cumsum function
result_values, result_mask = cumsum(values, mask, skipna=True)

print("\nAfter cumsum:")
print(f"  values (input):  {values}")
print(f"  result_values:   {result_values}")
print(f"  result_mask:     {result_mask}")

# Check if input was modified
if not np.array_equal(values, np.array([1, 2, 3, 4, 5])):
    print("\n❌ BUG CONFIRMED: Input array was modified!")
    print(f"   Original values: [1, 2, 3, 4, 5]")
    print(f"   Modified values: {values}")
else:
    print("\n✓ Input array was not modified")