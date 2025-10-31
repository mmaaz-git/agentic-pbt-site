import numpy as np
from xarray.indexes import RangeIndex

print("Reproducing the bug from the report:")
print("=" * 50)

start, stop, num = 817040.0, 0.0, 18

index = RangeIndex.linspace(start, stop, num, endpoint=True, dim="x")
values = index.transform.forward({"x": np.arange(num)})["x"]

print(f"Start: {start}")
print(f"Stop:  {stop}")
print(f"Num:   {num}")
print()
print(f"Last value: {values[-1]}")
print(f"Expected:   {stop}")
print(f"Match: {values[-1] == stop}")
print()
print(f"Difference: {values[-1] - stop}")
print(f"Relative error: {abs(values[-1] - stop) / max(abs(start), abs(stop))}")

# Compare with numpy
print("\nComparison with numpy.linspace:")
print("=" * 50)
numpy_comparison = np.linspace(start, stop, num, endpoint=True)
print(f"numpy.linspace last value: {numpy_comparison[-1]}")
print(f"numpy match: {numpy_comparison[-1] == stop}")

# Show all values
print("\nxarray values vs numpy values:")
print("-" * 50)
for i in range(num):
    xr_val = values[i]
    np_val = numpy_comparison[i]
    diff = xr_val - np_val
    print(f"i={i:2d}: xarray={xr_val:20.10f}, numpy={np_val:20.10f}, diff={diff:20.10e}")

# Test more cases
print("\n\nTesting additional cases:")
print("=" * 50)

test_cases = [
    (100.0, 0.0, 10),
    (1000.0, 0.0, 11),
    (10000.0, 0.0, 12),
    (-1000.0, 1000.0, 10),
    (0.0, 1.0, 10),
]

for start, stop, num in test_cases:
    index = RangeIndex.linspace(start, stop, num, endpoint=True, dim="x")
    values = index.transform.forward({"x": np.arange(num)})["x"]
    numpy_vals = np.linspace(start, stop, num, endpoint=True)

    xr_match = values[-1] == stop
    np_match = numpy_vals[-1] == stop
    diff = values[-1] - stop

    print(f"start={start:10.2f}, stop={stop:10.2f}, num={num:3d}: xr_match={xr_match}, np_match={np_match}, diff={diff:15.10e}")