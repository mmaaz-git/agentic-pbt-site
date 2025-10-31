import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from pandas.core.strings.accessor import cat_core, cat_safe

print("Testing cat_core with empty list:")
result_core = cat_core([], ',')
print(f'cat_core([], ","): {repr(result_core)}')
print(f'Type: {type(result_core).__name__}')
try:
    assert isinstance(result_core, np.ndarray), "Expected np.ndarray, got int"
    print("Assertion passed!")
except AssertionError as e:
    print(f"Assertion failed: {e}")

print("\nTesting cat_safe with empty list:")
result_safe = cat_safe([], ',')
print(f'cat_safe([], ","): {repr(result_safe)}')
print(f'Type: {type(result_safe).__name__}')
try:
    assert isinstance(result_safe, np.ndarray), "Expected np.ndarray, got int"
    print("Assertion passed!")
except AssertionError as e:
    print(f"Assertion failed: {e}")