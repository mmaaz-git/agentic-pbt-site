# Bug Report: numpy.char Case Conversion Functions Don't Handle Expanding Unicode Mappings

**Target**: `numpy.char.upper`, `numpy.char.lower`, `numpy.char.swapcase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The case conversion functions (`upper`, `lower`, `swapcase`) in `numpy.char` don't correctly handle Unicode characters that expand when case-converted. They claim to call `str.upper`/`str.lower`/`str.swapcase` element-wise but truncate the results to single characters.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np
import numpy.char as nc

@given(st.text(min_size=1))
def test_upper_matches_python(s):
    arr = np.array([s])
    numpy_result = nc.upper(arr)[0]
    python_result = s.upper()
    assert numpy_result == python_result
```

**Failing inputs**: `'ß'`, `'ﬁ'`, `'ﬂ'`, `'ﬀ'`, `'ﬃ'`, `'ﬄ'`, `'ﬅ'`, `'ﬆ'` for `upper`/`swapcase`; `'İ'` for `lower`

## Reproducing the Bug

```python
import numpy as np
import numpy.char as nc

arr = np.array(['ß'])
print(f"upper('ß'):    numpy={nc.upper(arr)[0]!r}, python={'ß'.upper()!r}")

arr = np.array(['ﬁ'])
print(f"upper('ﬁ'):    numpy={nc.upper(arr)[0]!r}, python={'ﬁ'.upper()!r}")

arr = np.array(['İ'])
print(f"lower('İ'):    numpy={nc.lower(arr)[0]!r}, python={'İ'.lower()!r}")

arr = np.array(['ß'])
print(f"swapcase('ß'): numpy={nc.swapcase(arr)[0]!r}, python={'ß'.swapcase()!r}")
```

Output:
```
upper('ß'):    numpy='S', python='SS'
upper('ﬁ'):    numpy='F', python='FI'
lower('İ'):    numpy='i', python='i̇'
swapcase('ß'): numpy='S', python='SS'
```

## Why This Is A Bug

The documentation for these functions explicitly states they call `str.upper`, `str.lower`, and `str.swapcase` element-wise. However, they truncate multi-character case conversion results to single characters, breaking compatibility with Python's string methods.

This affects:
- German text with 'ß' (sharp S)
- Text with typographic ligatures (ﬁ, ﬂ, ﬀ, etc.)
- Turkish text with 'İ' (uppercase I with dot)

## Fix

The underlying issue is that numpy string arrays have fixed-width character storage. When a character expands during case conversion, the array's dtype needs to accommodate the longer string. The fix would require:

1. Pre-calculating the maximum required string length after case conversion
2. Creating output arrays with sufficient dtype size
3. Or using variable-length string dtypes if available in newer numpy versions

This is a fundamental limitation of numpy's fixed-width string storage model and may not have a simple fix without changing how numpy.char handles string dtypes.