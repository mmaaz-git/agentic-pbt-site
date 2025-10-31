import pandas as pd
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env')
from xarray.core.indexes import PandasIndex

empty_idx = pd.Index([])
xr_idx = PandasIndex(empty_idx, "x")

print(f"Index length: {len(xr_idx.index)}")

try:
    rolled = xr_idx.roll({"x": 1})
    print("Successfully rolled empty index")
except ZeroDivisionError as e:
    print(f"ZeroDivisionError occurred: {e}")
    import traceback
    traceback.print_exc()