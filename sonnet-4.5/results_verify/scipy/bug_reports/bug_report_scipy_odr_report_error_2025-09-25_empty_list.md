# Bug Report: scipy.odr._report_error Returns Empty List

**Target**: `scipy.odr._odrpack._report_error`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_report_error` function returns an empty list for certain error codes (e.g., 10000, 20000, 30000), which violates the documented contract that it should return "A list of messages about why the odr() routine stopped."

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.odr._odrpack import _report_error

@settings(max_examples=100)
@given(
    digit1=st.integers(min_value=0, max_value=6),
    digit2=st.integers(min_value=0, max_value=9),
    digit3=st.integers(min_value=0, max_value=9),
    digit4=st.integers(min_value=0, max_value=9),
    digit5=st.integers(min_value=0, max_value=4)
)
def test_report_error_non_empty(digit1, digit2, digit3, digit4, digit5):
    info_code = digit1*10000 + digit2*1000 + digit3*100 + digit4*10 + digit5
    problems = _report_error(info_code)
    assert len(problems) >= 1
```

**Failing input**: `info_code=10000`

## Reproducing the Bug

```python
from scipy.odr._odrpack import _report_error

info_code = 10000
problems = _report_error(info_code)

print(f"info_code: {info_code}")
print(f"problems returned: {problems}")

assert len(problems) >= 1
```

## Why This Is A Bug

The function's docstring states it returns "A list of messages about why the odr() routine stopped." For error codes where `info >= 5` (indicating "questionable results or fatal error"), the function should always return at least one error message.

When `I[0] == 1` (codes 10000-19999) and all other digits are 0, the code only appends messages conditionally based on non-zero digits. If all are zero, `problems` remains empty, violating the documented behavior and making error handling impossible for callers.

## Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -143,12 +143,16 @@ def _report_error(info):
             problems.append(stopreason)
         elif I[0] == 1:
+            problems.append('Invalid input parameters')
             if I[1] != 0:
                 problems.append('N < 1')
             if I[2] != 0:
                 problems.append('M < 1')
             if I[3] != 0:
                 problems.append('NP < 1 or NP > N')
             if I[4] != 0:
                 problems.append('NQ < 1')
         elif I[0] == 2:
+            problems.append('Invalid dimension parameters')
             if I[1] != 0:
                 problems.append('LDY and/or LDX incorrect')
             if I[2] != 0:
@@ -158,6 +162,7 @@ def _report_error(info):
             if I[4] != 0:
                 problems.append('LWORK and/or LIWORK too small')
         elif I[0] == 3:
+            problems.append('Invalid scaling or step parameters')
             if I[1] != 0:
                 problems.append('STPB and/or STPD incorrect')
             if I[2] != 0: