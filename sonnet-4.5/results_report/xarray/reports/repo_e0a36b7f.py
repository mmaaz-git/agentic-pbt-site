from xarray.indexes import RangeIndex
import numpy as np

# Test the failing case
index = RangeIndex.arange(0.0, 1.5, 1.0, dim="x")

print(f"Expected step: 1.0")
print(f"Actual step: {index.step}")

coords = index.transform.forward({index.dim: np.arange(index.size)})
values = coords[index.coord_name]

print(f"Expected values: [0.0, 1.0]")
print(f"Actual values: {values}")

# Compare with numpy.arange
numpy_values = np.arange(0.0, 1.5, 1.0)
print(f"\nnumpy.arange(0.0, 1.5, 1.0) produces: {numpy_values}")
print(f"RangeIndex.arange produces different values: {values}")