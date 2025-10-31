# Bug Report: scipy.odr._report_error Returns Empty List

**Target**: `scipy.odr._odrpack._report_error`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_report_error` function returns an empty list when the info code has a ten-thousands digit (I[0]) of 7 or higher, violating its documented contract to always return a list of messages.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.integers(min_value=0, max_value=99999))
def test_report_error_returns_nonempty_list(info):
    from scipy.odr._odrpack import _report_error
    result = _report_error(info)
    assert isinstance(result, list), f"Expected list, got {type(result)}"
    assert len(result) > 0, "Expected non-empty list"
```

**Failing input**: `info=70000`

## Reproducing the Bug

```python
from scipy.odr._odrpack import _report_error

result = _report_error(70000)
print(f"Result: {result}")
print(f"Length: {len(result)}")

result2 = _report_error(80000)
print(f"Result for 80000: {result2}")

result3 = _report_error(4)
print(f"Result for 4 (valid): {result3}")
```

Output:
```
Result: []
Length: 0
Result for 80000: []
Result for 4 (valid): ['Iteration limit reached']
```

## Why This Is A Bug

The function's docstring states it returns "a list of messages about why the odr() routine stopped." For `info < 5`, it correctly returns `[stopreason]`. However, for `info >= 5` with ten-thousands digit >= 7, the if-elif chain at lines 136-176 doesn't match any condition, leaving the `problems` list empty.

This violates the API contract - the function should always return at least one message explaining what happened, even for unexpected error codes.

## Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -174,7 +174,9 @@ def _report_error(info):
             problems.append('Error occurred in callback')
         elif I[0] == 6:
             problems.append('Numerical error detected')
-
+        else:
+            problems.append(f'Unknown error (info={info})')
+            problems.append(stopreason)
         return problems

     else:
```