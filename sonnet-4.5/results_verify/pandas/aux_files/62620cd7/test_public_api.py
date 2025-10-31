import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import pandas as pd
import numpy as np

# Test if empty series causes issues
print("Testing with empty Series:")
s = pd.Series([], dtype=object)
print(f"Empty series: {s}")
print(f"s.str accessor available: {hasattr(s, 'str')}")

try:
    # Try concatenating empty series with separator
    result = s.str.cat(sep=',')
    print(f"s.str.cat(sep=','): {repr(result)}")
    print(f"Result type: {type(result).__name__}")
except Exception as e:
    print(f"Error: {e}")

print("\nTesting with empty Series and others:")
s2 = pd.Series([], dtype=object)
try:
    result = s.str.cat(s2, sep=',')
    print(f"s.str.cat(s2, sep=','): {repr(result)}")
    print(f"Result type: {type(result).__name__}")
except Exception as e:
    print(f"Error: {e}")