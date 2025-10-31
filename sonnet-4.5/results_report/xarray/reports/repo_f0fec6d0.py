import numpy as np
from xarray.indexes import RangeIndex

# Failing case from the bug report
start, stop, num = 817040.0, 0.0, 18

# Create xarray RangeIndex with endpoint=True
index = RangeIndex.linspace(start, stop, num, endpoint=True, dim="x")

# Get the actual values
values = index.transform.forward({"x": np.arange(num)})["x"]

print("=== XArray RangeIndex.linspace ===")
print(f"Start: {start}")
print(f"Stop: {stop}")
print(f"Number of points: {num}")
print(f"Endpoint: True")
print()
print(f"Last value generated: {values[-1]}")
print(f"Expected last value:  {stop}")
print(f"Exact match (values[-1] == stop): {values[-1] == stop}")
print(f"Error: {abs(values[-1] - stop)}")
print(f"Relative error: {abs((values[-1] - stop) / start) if start != 0 else 'N/A'}")
print()

# Compare with numpy.linspace
numpy_values = np.linspace(start, stop, num, endpoint=True)
print("=== NumPy linspace comparison ===")
print(f"NumPy last value: {numpy_values[-1]}")
print(f"NumPy exact match (last == stop): {numpy_values[-1] == stop}")
print()

# Show all values for context
print("=== All XArray values ===")
for i, val in enumerate(values):
    print(f"  [{i:2d}]: {val}")