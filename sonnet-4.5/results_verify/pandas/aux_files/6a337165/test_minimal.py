import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
import pyarrow as pa

# Test the minimal example from the bug report
lists = [[0, 1, 2, 3]]
pa_array = pa.array(lists, type=pa.list_(pa.int64()))
s = pd.Series(pa_array, dtype=pd.ArrowDtype(pa.list_(pa.int64())))

try:
    result = s.list[0:0]
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

# Also test what Python does with this slice
test_list = [0, 1, 2, 3]
print(f"\nPython list[0:0]: {test_list[0:0]}")
print(f"Python list[1:1]: {test_list[1:1]}")
print(f"Python list[2:2]: {test_list[2:2]}")