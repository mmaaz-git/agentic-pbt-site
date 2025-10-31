# Bug Report: numpy.char.replace() Truncates Results

**Target**: `numpy.char.replace()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.replace()` silently truncates results when the replacement would make the string longer than the original, causing silent data corruption.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=50))
@settings(max_examples=500, deadline=None)
def test_replace_matches_python(s):
    for old, new in [(s, s + 'x'), (s[0], s[0] * 2)]:
        numpy_result = char.replace(s, old, new)
        numpy_str = str(numpy_result.item() if hasattr(numpy_result, 'item') else numpy_result)
        python_result = s.replace(old, new)
        assert numpy_str == python_result
```

**Failing input**: `s='a'`, `old='a'`, `new='aa'`

## Reproducing the Bug

```python
import numpy.char as char

test_cases = [
    ('a', 'a', 'aa'),
    ('hello', 'hello', 'hello world'),
    ('test', 'test', 'testing'),
]

for haystack, old, new in test_cases:
    numpy_result = char.replace(haystack, old, new).item()
    python_result = haystack.replace(old, new)

    print(f"replace({repr(haystack)}, {repr(old)}, {repr(new)})")
    print(f"  numpy:  {repr(numpy_result)}")
    print(f"  python: {repr(python_result)}")

    if numpy_result != python_result:
        print(f"  MISMATCH!")
```

Output:
```
replace('a', 'a', 'aa')
  numpy:  'a'
  python: 'aa'
  MISMATCH!

replace('hello', 'hello', 'hello world')
  numpy:  'hello'
  python: 'hello world'
  MISMATCH!
```

## Why This Is A Bug

1. **Silent data corruption**: When replacement would expand the string, the result is truncated to the original length without any warning
2. **Violates documented behavior**: The docstring states "Calls `str.replace` element-wise", but the behavior differs from Python's `str.replace()`
3. **Breaks fundamental string operation**: String replacement is expected to work correctly regardless of the resulting length
4. **Affects real use cases**: Replacing abbreviated forms with full forms (e.g., 'Dr' → 'Doctor') would silently fail

## Fix

The issue stems from numpy's fixed-size character arrays. When `replace()` would produce a result longer than the original, the output array doesn't have sufficient space allocated.

The fix requires:
1. Pre-calculating the maximum possible result length based on counts of `old` in the input
2. Allocating output arrays with sufficient capacity: `new_len = len(s) + count(s, old) * (len(new) - len(old))`
3. Or, document this limitation and raise an error when truncation would occur

This likely requires changes in `numpy/_core/defchararray.py` or the underlying string ufunc implementation.