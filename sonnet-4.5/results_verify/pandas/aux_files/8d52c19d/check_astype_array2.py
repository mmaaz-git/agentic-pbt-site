#!/usr/bin/env python3
"""Check what astype_array returns - fixed"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.core.dtypes.astype import astype_array
import numpy as np

# Test what astype_array returns
arr = np.array([1, 2, 3])
result = astype_array(arr, dtype=np.dtype(np.float64), copy=False)
print(f"Input type: {type(arr)}")
print(f"Result type: {type(result)}")
print(f"Result: {result}")
print(f"Is ndarray: {isinstance(result, np.ndarray)}")