import numpy as np
from numpy.lib.array_utils import normalize_axis_tuple

# Test case demonstrating the bug
axis = 2_147_483_648  # This is 2^31, just beyond C int range
ndim = 1

try:
    result = normalize_axis_tuple(axis, ndim)
    print(f"No exception raised! Result: {result}")
except np.exceptions.AxisError as e:
    print(f"AxisError raised (expected): {e}")
except OverflowError as e:
    print(f"OverflowError raised (BUG!): {e}")
except Exception as e:
    print(f"Other exception raised: {type(e).__name__}: {e}")