# Bug Report: Cython.Plex.Regexps.Range IndexError on Odd-Length String

**Target**: `Cython.Plex.Regexps.Range`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Range` function crashes with `IndexError` when given an odd-length string in single-argument form, instead of validating the input and raising a proper error.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Plex.Regexps import Range

@given(st.text(min_size=1).filter(lambda s: len(s) % 2 == 1))
@settings(max_examples=200)
def test_range_validates_even_length(s):
    Range(s)
```

**Failing input**: `s='000'` (or any odd-length string)

## Reproducing the Bug

```python
from Cython.Plex.Regexps import Range

s = 'abc'
Range(s)
```

Output:
```
IndexError: string index out of range
```

## Why This Is A Bug

The `Range` function's docstring states: "Range(s) where |s| is a string of even length is an RE which matches any single character in the ranges |s[0]| to |s[1]|, |s[2]| to |s[3]|,..."

This clearly documents that the string must have even length. However, when given an odd-length string, the function crashes with `IndexError` at line 475:

```python
for i in range(0, len(s1), 2):
    ranges.append(CodeRange(ord(s1[i]), ord(s1[i + 1]) + 1))
```

When `len(s1)` is odd, the final iteration accesses `s1[i + 1]` when `i + 1 >= len(s1)`, causing an `IndexError`.

Users should receive a clear validation error (e.g., `PlexValueError`) explaining that the string must have even length, not a low-level `IndexError`.

## Fix

```diff
--- a/Cython/Plex/Regexps.py
+++ b/Cython/Plex/Regexps.py
@@ -471,6 +471,8 @@ def Range(s1, s2=None):
         result.str = "Range(%s,%s)" % (s1, s2)
     else:
+        if len(s1) % 2 != 0:
+            raise Errors.PlexValueError("Range() requires a string of even length, got length %d" % len(s1))
         ranges = []
         for i in range(0, len(s1), 2):
             ranges.append(CodeRange(ord(s1[i]), ord(s1[i + 1]) + 1))
```