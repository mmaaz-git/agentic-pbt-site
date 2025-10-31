import numpy as np
from pandas.core.array_algos.putmask import putmask_without_repeat

print("Reproducing the exact example from bug report:")
values = np.arange(10)
mask = np.ones(10, dtype=bool)
new = np.array([999])

print(f"values before: {values}")
print(f"mask: {mask}")
print(f"new: {new}")

putmask_without_repeat(values, mask, new)
print(f"values after: {values}")
print(f"Expected error but got result: all values are {values[0]}")