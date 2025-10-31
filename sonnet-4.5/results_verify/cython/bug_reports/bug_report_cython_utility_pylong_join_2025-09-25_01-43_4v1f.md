# Bug Report: Cython.Utility.pylong_join OverflowError with Large Negative Counts

**Target**: `Cython.Utility.pylong_join`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `pylong_join` function exhibits inconsistent behavior with negative count values: small negative values (e.g., -1, -2) return an empty string, but extremely large negative values (e.g., -4611686018427387905) trigger an OverflowError.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Utility import pylong_join

@given(st.integers(max_value=-1))
def test_negative_count(count):
    result = pylong_join(count)
    assert result == '', f"Negative count should produce empty string, got {repr(result)}"
```

**Failing input**: `count=-4611686018427387905`

## Reproducing the Bug

```python
from Cython.Utility import pylong_join

print("Small negative counts work:")
print(f"count=-1: {repr(pylong_join(-1))}")
print(f"count=-2: {repr(pylong_join(-2))}")

print("\nLarge negative count crashes:")
result = pylong_join(-4611686018427387905)
```

Output:
```
Small negative counts work:
count=-1: ''
count=-2: ''

Large negative count crashes:
OverflowError: cannot fit 'int' into an index-sized integer
```

## Why This Is A Bug

The function exhibits inconsistent behavior: it handles small negative counts by returning an empty string (via the empty range iteration), but crashes on large negative counts due to overflow in the string multiplication operation `'(' * (count * 2)`. While negative counts don't make semantic sense for this function, the behavior should be consistent - either all negative counts should be handled gracefully or all should raise a clear validation error.

## Fix

Add input validation to reject negative counts with a clear error message:

```diff
 def pylong_join(count, digits_ptr='digits', join_type='unsigned long'):
     """
     Generate an unrolled shift-then-or loop over the first 'count' digits.
     Assumes that they fit into 'join_type'.

     (((d[2] << n) | d[1]) << n) | d[0]
     """
+    if count < 0:
+        raise ValueError(f"count must be non-negative, got {count}")
     return ('(' * (count * 2) + ' | '.join(
         "(%s)%s[%d])%s)" % (join_type, digits_ptr, _i, " << PyLong_SHIFT" if _i else '')
         for _i in range(count-1, -1, -1)))
```