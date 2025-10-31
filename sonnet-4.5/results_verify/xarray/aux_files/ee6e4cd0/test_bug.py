import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from xarray.indexes import RangeIndex

# Test the specific case from the bug report
start = -1.5
stop = 0.0
step = 1.0

print("Testing xarray.indexes.RangeIndex.arange vs numpy.arange")
print(f"Input: start={start}, stop={stop}, step={step}")
print()

# NumPy behavior (expected)
np_range = np.arange(start, stop, step)
print(f"NumPy arange result: {np_range}")
print(f"NumPy size: {len(np_range)}")
print()

# XArray RangeIndex behavior
idx = RangeIndex.arange(start, stop, step, dim="x")
print(f"RangeIndex size: {idx.size}")
print(f"RangeIndex step (calculated): {idx.step}")
print()

# Generate the actual values from the RangeIndex
xr_values = idx.transform.forward({"x": np.arange(idx.size)})["x"]
print(f"RangeIndex values: {xr_values}")
print()

# Compare
print("Comparison:")
print(f"NumPy values:      {np_range}")
print(f"RangeIndex values: {xr_values}")
print(f"Are they equal? {np.allclose(xr_values, np_range)}")