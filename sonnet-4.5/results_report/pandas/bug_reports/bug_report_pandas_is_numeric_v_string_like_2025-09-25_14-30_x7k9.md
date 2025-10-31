# Bug Report: pandas.core.dtypes.common.is_numeric_v_string_like Not Symmetric

**Target**: `pandas.core.dtypes.common.is_numeric_v_string_like`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_numeric_v_string_like` function is not symmetric: `is_numeric_v_string_like(a, b)` can return a different value than `is_numeric_v_string_like(b, a)` for the same inputs. This violates the expected behavior for a comparison predicate and can lead to incorrect behavior when argument order varies.

## Property-Based Test

```python
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis import strategies as st
import numpy as np
from pandas.core.dtypes.common import is_numeric_v_string_like


@given(
    arrays(np.int64, shape=st.integers(min_value=1, max_value=10)),
    st.text(min_size=1, max_size=10)
)
@settings(max_examples=500)
def test_is_numeric_v_string_like_symmetric(arr, s):
    result1 = is_numeric_v_string_like(arr, s)
    result2 = is_numeric_v_string_like(s, arr)
    assert result1 == result2
```

**Failing input**: `arr=array([0]), s='0'`

## Reproducing the Bug

```python
import numpy as np
from pandas.core.dtypes.common import is_numeric_v_string_like

arr = np.array([0])
s = '0'

print(is_numeric_v_string_like(arr, s))
print(is_numeric_v_string_like(s, arr))
```

Output:
```
True
False
```

## Why This Is A Bug

The function is used to detect comparisons between numeric arrays and string-like objects that would trigger numpy deprecation warnings. Since comparison operations are commutative (e.g., `a == b` is the same as `b == a`), the function should be symmetric.

Looking at the implementation (lines 1025-1039 in common.py):

```python
is_a_array = isinstance(a, np.ndarray)
is_b_array = isinstance(b, np.ndarray)

is_a_numeric_array = is_a_array and a.dtype.kind in ("u", "i", "f", "c", "b")
is_b_numeric_array = is_b_array and b.dtype.kind in ("u", "i", "f", "c", "b")
is_a_string_array = is_a_array and a.dtype.kind in ("S", "U")
is_b_string_array = is_b_array and b.dtype.kind in ("S", "U")

is_b_scalar_string_like = not is_b_array and isinstance(b, str)

return (
    (is_a_numeric_array and is_b_scalar_string_like)
    or (is_a_numeric_array and is_b_string_array)
    or (is_b_numeric_array and is_a_string_array)
)
```

The function checks if `b` is a scalar string (`is_b_scalar_string_like` on line 1033) but never checks if `a` is a scalar string. This creates an asymmetry where:
- `is_numeric_v_string_like(numeric_array, scalar_string)` returns `True` (first clause)
- `is_numeric_v_string_like(scalar_string, numeric_array)` returns `False` (no matching clause)

The docstring examples show the function should work symmetrically:
```python
>>> is_numeric_v_string_like(np.array([1, 2]), np.array(["foo"]))
True
>>> is_numeric_v_string_like(np.array(["foo"]), np.array([1, 2]))
True
```

## Fix

```diff
--- a/pandas/core/dtypes/common.py
+++ b/pandas/core/dtypes/common.py
@@ -1030,10 +1030,12 @@ def is_numeric_v_string_like(a: ArrayLike, b) -> bool:
     is_a_string_array = is_a_array and a.dtype.kind in ("S", "U")
     is_b_string_array = is_b_array and b.dtype.kind in ("S", "U")

+    is_a_scalar_string_like = not is_a_array and isinstance(a, str)
     is_b_scalar_string_like = not is_b_array and isinstance(b, str)

     return (
         (is_a_numeric_array and is_b_scalar_string_like)
         or (is_a_numeric_array and is_b_string_array)
         or (is_b_numeric_array and is_a_string_array)
+        or (is_b_numeric_array and is_a_scalar_string_like)
     )
```