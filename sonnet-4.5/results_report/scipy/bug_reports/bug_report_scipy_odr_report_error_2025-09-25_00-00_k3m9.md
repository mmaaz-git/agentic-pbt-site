# Bug Report: scipy.odr._report_error Returns Empty List for High Info Values

**Target**: `scipy.odr._odrpack._report_error`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_report_error` function returns an empty list when the `info` parameter has a value >= 70000, violating its documented contract that it should return "A list of messages about why the odr() routine stopped."

## Property-Based Test

```python
from hypothesis import given, strategies as st
from scipy.odr._odrpack import _report_error

@given(st.integers(min_value=0, max_value=99999))
def test_report_error_always_returns_list(info):
    result = _report_error(info)
    assert len(result) > 0, f"_report_error({info}) returned empty list"
```

**Failing input**: `info=70000` (and any value where `info // 10000 % 10` >= 7)

## Reproducing the Bug

```python
from scipy.odr._odrpack import _report_error

result = _report_error(70000)
print(f"Result: {result}")
print(f"Length: {len(result)}")

assert len(result) > 0, "Expected non-empty list of error messages"
```

Output:
```
Result: []
Length: 0
AssertionError: Expected non-empty list of error messages
```

## Why This Is A Bug

1. **Violates documented contract**: The docstring states the function returns "A list of messages about why the odr() routine stopped." An empty list provides no information.

2. **Silent failure**: When used in `Output.pprint()` (line 619 of _odrpack.py), the code iterates over `self.stopreason`:
   ```python
   for r in self.stopreason:
       print(f'  {r}')
   ```
   An empty list causes no error output to be printed, leaving users confused about why the routine stopped.

3. **Incomplete error handling**: The function handles `I[0]` values 0-6 but silently returns an empty list for values 7-9.

## Fix

Add a default case to handle unknown error codes:

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -174,6 +174,8 @@ def _report_error(info):
             problems.append('Error occurred in callback')
         elif I[0] == 6:
             problems.append('Numerical error detected')
+        else:
+            problems.append(f'Unknown error code: {info}')

         return problems
```

This ensures the function always returns a non-empty list, providing at least minimal information about unexpected error codes.