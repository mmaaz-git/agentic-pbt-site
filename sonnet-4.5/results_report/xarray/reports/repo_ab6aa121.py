import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import numpy as np
from xarray.indexes import RangeIndex

start = -1.5
stop = 0.0
step = 1.0

# Create RangeIndex using arange
idx = RangeIndex.arange(start, stop, step, dim="x")

# Get the actual values from RangeIndex
xr_values = idx.transform.forward({"x": np.arange(idx.size)})["x"]

# Get the expected values from numpy.arange
np_values = np.arange(start, stop, step)

print(f"XArray values: {xr_values}")
print(f"NumPy values:  {np_values}")
print(f"XArray step: {idx.step}")
print(f"Expected step: {step}")
print(f"Discrepancy in second value: {xr_values[1]} (XArray) vs {np_values[1]} (NumPy)")