# Bug Report: numpy.char.replace() Truncates Results

**Target**: `numpy.char.replace()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.replace()` silently truncates results when the replacement would make the string longer than the original, causing data loss and incorrect output.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=50))
@settings(max_examples=500, deadline=None)
def test_replace_matches_python(s):
    old = s[0:1]
    new = s[0:1] * 2
    numpy_result = char.replace(s, old, new)
    numpy_str = str(numpy_result.item() if hasattr(numpy_result, 'item') else numpy_result)
    python_result = s.replace(old, new)
    assert numpy_str == python_result
```

**Failing input**: `s='a'`, where `old='a'` and `new='aa'`

## Reproducing the Bug

```python
import numpy.char as char

test_cases = [
    ('a', 'a', 'aa'),
    ('hello', 'hello', 'hello world'),
    ('test', 'test', 'testing'),
    ('straße', 'ß', 'ss'),
]

for haystack, old, new in test_cases:
    numpy_result = char.replace(haystack, old, new).item()
    python_result = haystack.replace(old, new)

    print(f"replace({repr(haystack)}, {repr(old)}, {repr(new)})")
    print(f"  numpy:  {repr(numpy_result)} (length {len(numpy_result)})")
    print(f"  python: {repr(python_result)} (length {len(python_result)})")

    if numpy_result != python_result:
        print(f"  TRUNCATED!")
```

Output:
```
replace('a', 'a', 'aa')
  numpy:  'a' (length 1)
  python: 'aa' (length 2)
  TRUNCATED!

replace('hello', 'hello', 'hello world')
  numpy:  'hello' (length 5)
  python: 'hello world' (length 11)
  TRUNCATED!
```

## Why This Is A Bug

1. **Silent data corruption**: Results are truncated without any warning or error
2. **Violates documented behavior**: Docstring states "Calls `str.replace` element-wise", but behavior differs from Python's `str.replace()`
3. **Breaks fundamental string operation**: String replacement should work correctly regardless of resulting length
4. **Affects common use cases**: Expanding abbreviations, adding prefixes/suffixes, or any replacement that increases string length

## Fix

The issue is that numpy allocates output arrays based on the original string length, not accounting for expansion from replacement.

The fix requires:
1. Pre-calculating result length: `new_len = len(s) + count(s, old) * (len(new) - len(old))`
2. Allocating output arrays with sufficient capacity before performing replacement
3. Or, documenting this limitation and raising an error when truncation would occur

This requires changes in `numpy/_core/defchararray.py` in the `replace()` function implementation.