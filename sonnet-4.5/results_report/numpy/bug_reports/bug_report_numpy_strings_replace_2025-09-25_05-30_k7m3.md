# Bug Report: numpy.strings.replace Truncates Replacement String

**Target**: `numpy.strings.replace`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When replacing a substring with a longer string in an array with a small dtype (e.g., `<U1`), the replacement string is truncated to fit the input array's dtype before the replacement operation, causing incorrect results.

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import given, strategies as st, assume, settings

def string_arrays():
    return st.lists(
        st.text(
            alphabet=st.characters(min_codepoint=32, max_codepoint=126),
            min_size=0, max_size=50
        ),
        min_size=1, max_size=20
    ).map(lambda lst: np.array(lst, dtype='U'))

def simple_strings():
    return st.text(
        alphabet=st.characters(min_codepoint=32, max_codepoint=126),
        min_size=0, max_size=20
    )

@given(string_arrays(), simple_strings(), simple_strings())
@settings(max_examples=1000)
def test_replace_length_calculation(arr, old_str, new_str):
    assume(old_str != '')

    result = nps.replace(arr, old_str, new_str)

    original_lengths = nps.str_len(arr)
    result_lengths = nps.str_len(result)
    counts = nps.count(arr, old_str)
    expected_lengths = original_lengths + counts * (len(new_str) - len(old_str))

    assert np.array_equal(result_lengths, expected_lengths)
```

**Failing input**: `arr=['0']`, `old_str='0'`, `new_str='00'`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

arr = np.array(['0'], dtype='U')
result = nps.replace(arr, '0', '00')

print(f"Result: {result}")
print(f"Expected: ['00']")
print(f"Got: ['{result[0]}']")

assert str(result[0]) == '00'
```

Output:
```
Result: ['0']
Expected: ['00']
Got: ['0']
AssertionError: assert '0' == '00'
```

## Why This Is A Bug

The function violates Python's `str.replace` semantics and produces incorrect results. Python's `'0'.replace('0', '00')` correctly returns `'00'`, but NumPy's vectorized version returns `'0'`. This bug occurs because the replacement string is cast to the input array's dtype (`<U1`) on line 1359, truncating `'00'` to `'0'` before the replacement operation.

## Fix

The issue is on line 1359 where `new` is cast to the input array's dtype. The replacement string should not be constrained by the input array's dtype, as the output array is sized correctly to accommodate longer strings.

```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -1356,7 +1356,10 @@ def replace(a, old, new, count=-1):

     a_dt = arr.dtype
     old = old.astype(old_dtype or a_dt, copy=False)
-    new = new.astype(new_dtype or a_dt, copy=False)
+    # Don't constrain 'new' to input dtype - it can be longer
+    if new_dtype is None:
+        new = new.astype(f"{a_dt.char}{str_len(new).max()}", copy=False)
+
     max_int64 = np.iinfo(np.int64).max
     counts = _count_ufunc(arr, old, 0, max_int64)
     counts = np.where(count < 0, counts, np.minimum(counts, count))
```