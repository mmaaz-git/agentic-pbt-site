import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
from pandas import MultiIndex

# Test what MultiIndex.from_tuples does with an empty list
print("Testing MultiIndex.from_tuples with empty list:")
try:
    mi = MultiIndex.from_tuples([])
    print(f"Success! Created MultiIndex: {mi}")
    print(f"  nlevels: {mi.nlevels}")
    print(f"  shape: {mi.shape}")
    print(f"  codes: {mi.codes}")
    print(f"  levels: {mi.levels}")
except Exception as e:
    print(f"Failed with {type(e).__name__}: {e}")