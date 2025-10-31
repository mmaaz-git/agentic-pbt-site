import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from pandas.core.dtypes.common import ensure_python_int

print("Testing ensure_python_int with 5.5:")
try:
    result = ensure_python_int(5.5)
    print(f"Unexpected: got {result}")
except TypeError as e:
    print(f"TypeError for 5.5: {e}")

print("\nTesting ensure_python_int with large int as float:")
try:
    large_int = 9007199254740993
    result = ensure_python_int(np.float64(large_int))
    print(f"Unexpected: got {result}")
except TypeError as e:
    print(f"TypeError for large int as float: {e}")

print("\nTesting ensure_python_int with 5.0 (integer-valued float):")
try:
    result = ensure_python_int(5.0)
    print(f"Got result: {result}, type: {type(result)}")
except TypeError as e:
    print(f"TypeError for 5.0: {e}")

print("\nTesting ensure_python_int with np.float64(5.0):")
try:
    result = ensure_python_int(np.float64(5.0))
    print(f"Got result: {result}, type: {type(result)}")
except TypeError as e:
    print(f"TypeError for np.float64(5.0): {e}")