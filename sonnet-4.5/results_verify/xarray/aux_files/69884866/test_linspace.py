#!/usr/bin/env python3
"""Compare linspace and arange implementations"""

from xarray.indexes import RangeIndex
import numpy as np

# Test linspace
print("=== Testing linspace ===")
index = RangeIndex.linspace(0.0, 1.0, 5, endpoint=True, dim="x")
print(f"Linspace index.start: {index.start}")
print(f"Linspace index.stop: {index.stop}")
print(f"Linspace index.step: {index.step}")
print(f"Linspace index.size: {index.size}")

coords = index.transform.forward({index.dim: np.arange(index.size)})
values = coords[index.coord_name]
print(f"Linspace values: {values}")

# Compare with numpy.linspace
numpy_values = np.linspace(0.0, 1.0, 5, endpoint=True)
print(f"numpy.linspace values: {numpy_values}")
print(f"Values match: {np.allclose(values, numpy_values)}")

# Check the internal storage
print(f"\nInternal representation:")
print(f"Transform.start: {index.transform.start}")
print(f"Transform.stop: {index.transform.stop}")
print(f"Transform.size: {index.transform.size}")

# Look at the linspace implementation more closely
print("\n=== Linspace implementation detail ===")
start = 0.0
stop = 1.0
num = 5
endpoint = True

if endpoint:
    adjusted_stop = stop + (stop - start) / (num - 1)
    print(f"Adjusted stop for endpoint=True: {adjusted_stop}")
else:
    adjusted_stop = stop

print(f"This stores: start={start}, stop={adjusted_stop}, size={num}")
print(f"Step becomes: ({adjusted_stop} - {start}) / {num} = {(adjusted_stop - start) / num}")