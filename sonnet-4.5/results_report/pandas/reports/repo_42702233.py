import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
import pyarrow as pa

# Create a Series with a list containing integers
lists = [[0, 1, 2, 3]]
pa_array = pa.array(lists, type=pa.list_(pa.int64()))
s = pd.Series(pa_array, dtype=pd.ArrowDtype(pa.list_(pa.int64())))

print("Original Series:")
print(s)
print()

print("Attempting to perform empty slice s.list[0:0]...")
try:
    result = s.list[0:0]
    print("Result:")
    print(result)
except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")