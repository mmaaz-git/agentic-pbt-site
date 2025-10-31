import numpy as np
from pandas.core.array_algos.putmask import putmask_without_repeat

# Create test data
values = np.arange(10)
mask = np.ones(10, dtype=bool)
new = np.array([999])

print("Before putmask_without_repeat:")
print(f"  values: {values}")
print(f"  mask: {mask}")
print(f"  new: {new}")
print(f"  Length of values: {len(values)}")
print(f"  Number of True values in mask: {mask.sum()}")
print(f"  Length of new: {len(new)}")

# This should raise ValueError according to documentation
# but it doesn't when new has length 1
putmask_without_repeat(values, mask, new)

print("\nAfter putmask_without_repeat:")
print(f"  values: {values}")
print("\nExpected: ValueError('cannot assign mismatch length to masked array')")
print("Actual: No error raised, single value repeated across all positions")