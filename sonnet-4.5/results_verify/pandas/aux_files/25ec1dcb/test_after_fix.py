import sys
import numpy as np

# Force Python to reimport the module
if 'pandas.core.array_algos.take' in sys.modules:
    del sys.modules['pandas.core.array_algos.take']

import pandas.core.array_algos.take as take_module

arr = np.array([1, 2, 3, 4, 5], dtype=np.int32)
indexer = np.array([0, 1, 2], dtype=np.intp)
mask_all_false = np.array([False, False, False], dtype=bool)

result_with_mask = take_module.take_1d(arr, indexer, fill_value=np.nan, allow_fill=True, mask=mask_all_false)
result_without_mask = take_module.take_1d(arr, indexer, fill_value=np.nan, allow_fill=True, mask=None)

print(f"With explicit all-False mask: dtype={result_with_mask.dtype}")
print(f"With mask=None: dtype={result_without_mask.dtype}")

try:
    assert result_with_mask.dtype == result_without_mask.dtype, \
        f"Inconsistent dtypes: {result_with_mask.dtype} vs {result_without_mask.dtype}"
    print("SUCCESS: Both dtypes are now consistent!")
except AssertionError as e:
    print(f"FAILED: {e}")