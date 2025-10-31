# Bug Report: Cython.Plex Range() Unhelpful Error for Odd-Length Strings

**Target**: `Cython.Plex.Range`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Range()` function raises an unhelpful `IndexError: string index out of range` when given an odd-length string, instead of a clear error message explaining the input requirement.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pytest
from Cython.Plex import Range


@settings(max_examples=100)
@given(st.text(min_size=1, max_size=20))
def test_range_validates_input_properly(s):
    if len(s) % 2 == 0:
        re = Range(s)
        assert hasattr(re, 'nullable')
    else:
        with pytest.raises(ValueError, match="even length"):
            Range(s)
```

**Failing input**: `s='abc'` (any odd-length string)

## Reproducing the Bug

```python
from Cython.Plex import Range

Range('abc')
```

Output:
```
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "Cython/Plex/Regexps.py", line X, in Range
    ranges.append(CodeRange(ord(s1[i]), ord(s1[i + 1]) + 1))
                                        ~~~^^^^^^^
IndexError: string index out of range
```

## Why This Is A Bug

The function's docstring explicitly states:
> Range(s) where |s| is a string of even length is an RE which matches any single character in the ranges |s[0]| to |s[1]|, |s[2]| to |s[3]|,...

When this precondition is violated, the function should raise a clear, actionable error like:
```python
ValueError: Range() requires a string of even length, got length 3
```

Instead, it raises a generic `IndexError` that doesn't explain what went wrong or how to fix it. This violates the principle of helpful error messages and makes the API harder to use correctly.

## Fix

```diff
def Range(s1, s2=None):
    """
    Range(c1, c2) is an RE which matches any single character in the range
    |c1| to |c2| inclusive.
    Range(s) where |s| is a string of even length is an RE which matches
    any single character in the ranges |s[0]| to |s[1]|, |s[2]| to |s[3]|,...
    """
    if s2:
        result = CodeRange(ord(s1), ord(s2) + 1)
        result.str = "Range(%s,%s)" % (s1, s2)
    else:
+       if len(s1) % 2 != 0:
+           raise ValueError(f"Range() requires a string of even length, got length {len(s1)}")
        ranges = []
        for i in range(0, len(s1), 2):
            ranges.append(CodeRange(ord(s1[i]), ord(s1[i + 1]) + 1))
        result = Alt(*ranges)
        result.str = "Range(%s)" % repr(s1)
    return result
```