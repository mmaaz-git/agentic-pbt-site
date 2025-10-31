import math
import numpy as np
from xarray.indexes import RangeIndex

# Create RangeIndex with incompatible direction
index = RangeIndex.arange(start=1.0, stop=0.0, step=1.0, dim="x")
print(f"Size: {index.size}")
print(f"Start: {index.start}, Stop: {index.stop}, Step: {index.step}")

# Show how the size is computed
size_computed = math.ceil((0.0 - 1.0) / 1.0)
print(f"\nSize computed as: math.ceil((stop - start) / step) = math.ceil(({0.0} - {1.0}) / {1.0}) = {size_computed}")

# Compare with NumPy's behavior
print(f"\nNumPy's behavior:")
arr = np.arange(1.0, 0.0, 1.0)
print(f"np.arange(1.0, 0.0, 1.0) = {arr}")
print(f"np.arange(1.0, 0.0, 1.0).size = {arr.size}")

# Show that negative size is problematic
print(f"\nProblems with negative size:")
print(f"- A negative dimension size violates array constraints")
print(f"- Could cause downstream errors in operations expecting non-negative sizes")
print(f"- Deviates from NumPy's established behavior (returns empty array)")