import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import dask.array as da
from xarray.compat.dask_array_compat import reshape_blockwise

arr = da.arange(1).reshape(1, 1)
result = reshape_blockwise(arr, (1,))

print(f"Expected: (1,)")
print(f"Got: {result.shape}")
assert result.shape == (1,), f"Expected (1,), got {result.shape}"