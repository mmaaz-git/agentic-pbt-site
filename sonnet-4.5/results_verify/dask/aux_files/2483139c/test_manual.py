import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import numpy as np
import dask.array as da

arr = np.array([0])
darr = da.from_array(arr, chunks=-1)

idx = slice(-2, -2, -1)

np_result = arr[idx]
dask_result = darr[idx].compute()

print(f"NumPy: arr[{idx}] = {np_result}")
print(f"Dask:  darr[{idx}].compute() = {dask_result}")

try:
    assert np.array_equal(np_result, dask_result)
    print("Assertion passed: Results are equal")
except AssertionError:
    print("AssertionError: Results are NOT equal!")
    print(f"  NumPy returned empty array: {len(np_result) == 0}")
    print(f"  Dask returned: {dask_result}")