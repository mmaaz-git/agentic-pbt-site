# Bug Report: numpy.strings.capitalize/title Null Character Removal

**Target**: `numpy.strings.capitalize`, `numpy.strings.title`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.strings.capitalize` and `numpy.strings.title` incorrectly remove null characters when the null character is the only character in the string, differing from Python's behavior.

## Property-Based Test

```python
import numpy as np
from hypothesis import given, strategies as st

@given(st.text())
def test_capitalize_matches_python(s):
    arr = np.array([s])
    numpy_result = np.strings.capitalize(arr)[0]
    python_result = s.capitalize()
    assert numpy_result == python_result
```

**Failing input**: `'\x00'`

## Reproducing the Bug

```python
import numpy as np

s = '\x00'
arr = np.array([s])

cap_result = np.strings.capitalize(arr)[0]
title_result = np.strings.title(arr)[0]

print(f"Input: {s!r}")
print(f"Python capitalize: {s.capitalize()!r}")
print(f"NumPy capitalize:  {cap_result!r}")
print(f"Python title: {s.title()!r}")
print(f"NumPy title:  {title_result!r}")
```

**Output**:
```
Input: '\x00'
Python capitalize: '\x00'
NumPy capitalize:  ''
Python title: '\x00'
NumPy title:  ''
```

## Why This Is A Bug

Both functions claim to call the corresponding Python string methods element-wise (lines 1224 and 1267), but NumPy removes null characters when they appear alone in a string. While null characters are rare in practice, the behavior violates the documented contract.

Note: When the null character appears with other characters (e.g., 'a\x00b'), NumPy correctly preserves it.

## Fix

The issue is in lines 1255 and 1298 where `_vec_string` is called. The underlying implementation likely treats null-only strings specially, possibly trimming them. The fix would require ensuring that `_vec_string` preserves all characters including nulls, even when they're the only character in the string.