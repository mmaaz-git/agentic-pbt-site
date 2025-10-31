# Bug Report: Null Byte Removal in String Operations

**Target**: `numpy.strings` (multiple functions)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Multiple `numpy.strings` functions incorrectly remove standalone null bytes (`\x00`), treating them as empty strings instead of preserving them like Python's string methods do.

## Property-Based Test

```python
import numpy as np
import numpy.strings as ns
from hypothesis import given, strategies as st


@given(st.lists(st.just('\x00'), min_size=1))
def test_upper_preserves_null_bytes(strings):
    arr = np.array(strings, dtype=np.str_)
    result = ns.upper(arr)

    for orig, res in zip(strings, result):
        expected = orig.upper()
        assert res == expected
```

**Failing input**: `strings=['\x00']`

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as ns

arr = np.array(['\x00'], dtype=np.str_)

print(f"upper: numpy={repr(ns.upper(arr)[0])}, python={repr('\x00'.upper())}")
print(f"lower: numpy={repr(ns.lower(arr)[0])}, python={repr('\x00'.lower())}")
print(f"capitalize: numpy={repr(ns.capitalize(arr)[0])}, python={repr('\x00'.capitalize())}")
print(f"title: numpy={repr(ns.title(arr)[0])}, python={repr('\x00'.title())}")
print(f"swapcase: numpy={repr(ns.swapcase(arr)[0])}, python={repr('\x00'.swapcase())}")
print(f"strip: numpy={repr(ns.strip(arr)[0])}, python={repr('\x00'.strip())}")
print(f"lstrip: numpy={repr(ns.lstrip(arr)[0])}, python={repr('\x00'.lstrip())}")
print(f"rstrip: numpy={repr(ns.rstrip(arr)[0])}, python={repr('\x00'.rstrip())}")

left, mid, right = ns.partition(arr, 'X')
print(f"partition: numpy=({repr(left[0])}, {repr(mid[0])}, {repr(right[0])}), python={repr('\x00'.partition('X'))}")
```

**Output:**
```
upper: numpy='', python='\x00'
lower: numpy='', python='\x00'
capitalize: numpy='', python='\x00'
title: numpy='', python='\x00'
swapcase: numpy='', python='\x00'
strip: numpy='', python='\x00'
lstrip: numpy='', python='\x00'
rstrip: numpy='', python='\x00'
partition: numpy=('', '', ''), python=('\x00', '', '')
```

## Why This Is A Bug

All affected functions claim to replicate Python string methods element-wise. Python preserves null bytes in all operations:
- `'\x00'.upper()` returns `'\x00'`
- `'hel\x00lo'.upper()` returns `'HEL\x00LO'`

However, numpy.strings functions incorrectly convert standalone null bytes to empty strings. This is inconsistent because:
1. Null bytes in the middle of strings ARE correctly preserved
2. Only when a string consists entirely of `\x00` is it converted to `''`
3. This changes string length (`len('\x00') = 1`, but `len(result) = 0`)

This suggests the C implementation treats a null-byte-only string as a C string terminator or empty string.

**Affected functions**: `upper`, `lower`, `capitalize`, `title`, `swapcase`, `strip`, `lstrip`, `rstrip`, `partition`, `rpartition`

## Fix

The fix requires changes to the C/ufunc implementation to:
- Properly handle strings consisting only of `\x00`
- Not use null bytes as string terminators
- Maintain consistency with how null bytes are handled within longer strings

This is likely a systematic issue in the string ufunc implementation that affects all string transformation operations.