# Bug Report: scipy.io.matlab Null Character Handling in Strings

**Target**: `scipy.io.matlab` (savemat/loadmat)
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The savemat/loadmat round-trip corrupts strings containing null characters (`\x00`) by replacing them with spaces, violating the documented round-trip property.

## Property-Based Test

```python
from io import BytesIO
import numpy as np
from hypothesis import given, strategies as st, settings
from scipy.io.matlab import loadmat, savemat


@settings(max_examples=100)
@given(
    st.text(alphabet=st.characters(min_codepoint=1, max_codepoint=127), min_size=1, max_size=50),
    st.integers(min_value=0, max_value=10),
)
def test_string_with_embedded_nulls(prefix, num_nulls):
    s = prefix + '\x00' * num_nulls
    bio = BytesIO()
    data = {"s": s}
    savemat(bio, data)
    bio.seek(0)
    loaded = loadmat(bio)

    if loaded["s"].size > 0:
        result = loaded["s"].flatten()
        if result.dtype.kind == 'U':
            reconstructed = result[0]
        else:
            reconstructed = ''.join(result)

        assert reconstructed == s
```

**Failing input**: `prefix='0', num_nulls=1` (produces string `'0\x00'`)

## Reproducing the Bug

```python
from io import BytesIO
from scipy.io.matlab import loadmat, savemat

test_string = '0\x00'
print(f"Original: {repr(test_string)}")

bio = BytesIO()
savemat(bio, {"s": test_string})
bio.seek(0)
loaded = loadmat(bio)

result = loaded["s"][0][0]
print(f"Loaded:   {repr(result)}")
print(f"Match:    {result == test_string}")
```

**Output:**
```
Original: '0\x00'
Loaded:   '0 '
Match:    False
```

## Why This Is A Bug

1. The `loadmat` docstring explicitly claims: "The default setting is True, because it allows easier round-trip load and save of MATLAB files."

2. Null characters (`\x00`) are valid Python string characters and should be preserved during serialization.

3. The corruption happens silently without any warning, causing data loss.

## Fix

The bug originates in `_miobase.py` in the `arr_to_chars` function at lines 433-434:

```python
arr = arr.copy()
arr[tuple(empties)] = ' '
```

This code replaces what it considers "empty" characters with spaces. The issue is on line 430:

```python
empties = [arr == np.array('', dtype=arr.dtype)]
```

This comparison treats null characters as empty strings. The fix should distinguish between truly empty strings and null characters. One approach:

```diff
--- a/scipy/io/matlab/_miobase.py
+++ b/scipy/io/matlab/_miobase.py
@@ -427,9 +427,11 @@ def arr_to_chars(arr):
     arr = np.ndarray(shape=dims,
                      dtype=arr_dtype_number(arr, 1),
                      buffer=arr)
-    empties = [arr == np.array('', dtype=arr.dtype)]
-    if not np.any(empties):
-        return arr
-    arr = arr.copy()
-    arr[tuple(empties)] = ' '
+    # Check for empty strings but exclude null characters
+    # Null characters (\x00) should be preserved, not replaced
+    empties = [(arr == np.array('', dtype=arr.dtype)) & (arr.view('u1').reshape(arr.shape + (-1,)).sum(axis=-1) == 0)]
+    if np.any(empties):
+        arr = arr.copy()
+        # Only replace truly empty strings, not null characters
+        arr[tuple(empties)] = ' '
     return arr
```

Alternatively, the replacement behavior might need reconsideration entirely - it's unclear why empty characters should be replaced with spaces in the first place.