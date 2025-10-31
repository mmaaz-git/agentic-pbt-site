# Bug Report: Cython.Runtime.refnanny.Context Integer Overflow

**Target**: `Cython.Runtime.refnanny.Context.__init__`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `Context` constructor crashes with `OverflowError` when the `line` parameter exceeds the maximum value representable by a C `ssize_t`, instead of validating input or handling the overflow gracefully.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings

@given(st.integers())
@settings(max_examples=200)
def test_line_number_range(line):
    from Cython.Runtime import refnanny

    ctx = refnanny.Context("test", line)
    assert ctx.start == line
```

**Failing input**: `line=9_223_372_036_854_775_808`

## Reproducing the Bug

```python
from Cython.Runtime import refnanny

line = 9_223_372_036_854_775_808
ctx = refnanny.Context("test", line)
```

Output:
```
OverflowError: Python int too large to convert to C ssize_t
```

## Why This Is A Bug

The `Context` constructor accepts a Python `int` parameter for `line` but doesn't validate that it fits within the C `ssize_t` range. This causes an unhandled `OverflowError` instead of either:
1. Validating the input and raising a more informative error, or
2. Clamping the value to the valid range, or
3. Documenting the constraint in the function signature/docstring

While extremely large line numbers are rare in practice, the API should handle them gracefully rather than crashing.

## Fix

Add input validation in the `__cinit__` method to check if the line number is within valid bounds:

```diff
--- a/Cython/Runtime/refnanny.pyx
+++ b/Cython/Runtime/refnanny.pyx
@@ -55,6 +55,10 @@ cdef class Context(object):
     cdef __Pyx_refnanny_mutex lock

     def __cinit__(self, name, line=0, filename=None):
+        import sys
+        if not (-sys.maxsize - 1 <= line <= sys.maxsize):
+            raise ValueError(f"line number {line} out of valid range "
+                           f"[{-sys.maxsize - 1}, {sys.maxsize}]")
         self.name = name
         self.start = line
         self.filename = filename
```