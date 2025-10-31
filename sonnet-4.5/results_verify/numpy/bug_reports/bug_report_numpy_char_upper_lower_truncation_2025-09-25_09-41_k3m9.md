# Bug Report: numpy.char.upper() and char.lower() Silently Truncate Unicode Case Conversions

**Target**: `numpy.char.upper()` and `numpy.char.lower()`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.upper()` and `char.lower()` silently truncate the results when Unicode case conversion produces strings longer than the input array's dtype can hold, causing data loss without warning.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st, settings, assume

unicode_strat = st.text(alphabet=st.characters(min_codepoint=0x80, max_codepoint=0x10000), min_size=1, max_size=30)


@given(unicode_strat)
@settings(max_examples=300)
def test_upper_lower_unicode(s):
    assume(not s.endswith('\x00'))

    arr = np.array([s])

    upper = char.upper(arr)[0]
    lower = char.lower(arr)[0]

    expected_upper = s.upper()
    expected_lower = s.lower()

    assert upper == expected_upper, f"upper unicode mismatch: {upper!r} != {expected_upper!r}"
    assert lower == expected_lower, f"lower unicode mismatch: {lower!r} != {expected_lower!r}"
```

**Failing inputs**:
- `s='ß'` for upper (German sharp s)
- `s='ŉ'` for upper (Latin small n preceded by apostrophe)
- `s='İ'` for lower (Latin capital I with dot above)

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

s = 'ß'
arr = np.array([s])

print(f"Python: {s!r}.upper() = {s.upper()!r}")
print(f"NumPy: char.upper({s!r}) = {char.upper(arr)[0]!r}")

assert s.upper() == 'SS'
assert char.upper(arr)[0] == 'S'
```

**Output**:
```
Python: 'ß'.upper() = 'SS'
NumPy: char.upper('ß') = 'S'
```

Additional failing cases:
- `'ŉ'.upper()`: Python `'ʼN'` (2 chars) → NumPy `'ʼ'` (1 char)
- `'İ'.lower()`: Python `'i̇'` (2 chars with combining dot) → NumPy `'i'` (1 char)

## Why This Is A Bug

1. The docstring for `char.upper()` states it "calls str.upper element-wise" with no mention of truncation
2. Silent data loss violates user expectations for string operations
3. `np.array(['ß'], dtype='U10')` works correctly, showing the issue is dtype inference, not fundamental limitation
4. No warning or error is raised when truncation occurs

This breaks the fundamental property that `char.upper(arr)` should behave like Python's `str.upper()` applied element-wise.

## Fix

The fix should make `char.upper()` and `char.lower()` automatically expand the output dtype when needed:

```diff
--- a/numpy/_core/defchararray.py
+++ b/numpy/_core/defchararray.py
@@ -upper_function
-    return _vec_string(a, a.dtype.type, 'upper')
+    # Calculate max output size for case conversion
+    # Some Unicode chars expand (ß->SS, ŉ->ʼN, İ->i̇)
+    result = _vec_string(a, np.object_, 'upper')
+    # Find max string length in result
+    max_len = max(len(str(x)) for x in result.flat)
+    # Create appropriately-sized output dtype
+    out_dtype = f'U{max_len}'
+    return np.asarray(result, dtype=out_dtype)
```

Alternatively, raise a clear error when truncation would occur rather than silently losing data.