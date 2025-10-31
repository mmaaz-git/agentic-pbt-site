import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from pandas.core.strings.accessor import cat_core, cat_safe

result_core = cat_core([], ',')
print(f'cat_core([], ","): {repr(result_core)}')
print(f'Type: {type(result_core).__name__}')
print(f'Is np.ndarray?: {isinstance(result_core, np.ndarray)}')

try:
    assert isinstance(result_core, np.ndarray), "Expected np.ndarray, got int"
except AssertionError as e:
    print(f'AssertionError for cat_core: {e}')

print()

result_safe = cat_safe([], ',')
print(f'cat_safe([], ","): {repr(result_safe)}')
print(f'Type: {type(result_safe).__name__}')
print(f'Is np.ndarray?: {isinstance(result_safe, np.ndarray)}')

try:
    assert isinstance(result_safe, np.ndarray), "Expected np.ndarray, got int"
except AssertionError as e:
    print(f'AssertionError for cat_safe: {e}')