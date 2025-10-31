# Bug Report: scipy.odr._odrpack._report_error Returns Empty List for Multiple Error Codes

**Target**: `scipy.odr._odrpack._report_error`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `_report_error` function returns an empty list for multiple `info` values, violating its documented contract that promises "A list of messages about why the odr() routine stopped."

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from scipy.odr._odrpack import _report_error

@given(st.integers(min_value=0, max_value=99999))
@settings(max_examples=100)
def test_report_error_always_returns_list(info):
    """Test that _report_error always returns a non-empty list as documented."""
    result = _report_error(info)
    assert len(result) > 0, f"_report_error({info}) returned empty list"

if __name__ == "__main__":
    # Run the test
    test_report_error_always_returns_list()
```

<details>

<summary>
**Failing input**: `info=70000`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 13, in <module>
    test_report_error_always_returns_list()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 5, in test_report_error_always_returns_list
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 9, in test_report_error_always_returns_list
    assert len(result) > 0, f"_report_error({info}) returned empty list"
           ^^^^^^^^^^^^^^^
AssertionError: _report_error(70000) returned empty list
Falsifying example: test_report_error_always_returns_list(
    info=70000,
)
```
</details>

## Reproducing the Bug

```python
from scipy.odr._odrpack import _report_error

# Test the problematic value
result = _report_error(70000)
print(f"Result for info=70000: {result}")
print(f"Length: {len(result)}")

# Test other high values
for info in [70000, 80000, 90000]:
    result = _report_error(info)
    print(f"\ninfo={info}")
    print(f"  Result: {result}")
    print(f"  Length: {len(result)}")

# Test to understand the pattern
print("\n--- Understanding the pattern ---")
for info in [0, 1, 2, 3, 4, 5, 60000, 70000]:
    result = _report_error(info)
    I0 = (info // 10000) % 10
    print(f"info={info:5d}, I[0]={I0}, len(result)={len(result)}, result={result}")

# Assert that should fail for the bug
print("\n--- Testing assertion ---")
try:
    result = _report_error(70000)
    assert len(result) > 0, f"Expected non-empty list of error messages, got: {result}"
    print("Assertion passed")
except AssertionError as e:
    print(f"AssertionError: {e}")
```

<details>

<summary>
Empty list returned for multiple error codes
</summary>
```
Result for info=70000: []
Length: 0

info=70000
  Result: []
  Length: 0

info=80000
  Result: []
  Length: 0

info=90000
  Result: []
  Length: 0

--- Understanding the pattern ---
info=    0, I[0]=0, len(result)=1, result=['Blank']
info=    1, I[0]=0, len(result)=1, result=['Sum of squares convergence']
info=    2, I[0]=0, len(result)=1, result=['Parameter convergence']
info=    3, I[0]=0, len(result)=1, result=['Both sum of squares and parameter convergence']
info=    4, I[0]=0, len(result)=1, result=['Iteration limit reached']
info=    5, I[0]=0, len(result)=1, result=['Blank']
info=60000, I[0]=6, len(result)=1, result=['Numerical error detected']
info=70000, I[0]=7, len(result)=0, result=[]

--- Testing assertion ---
AssertionError: Expected non-empty list of error messages, got: []
```
</details>

## Why This Is A Bug

1. **Violates documented contract**: The docstring explicitly states the function returns "A list of messages about why the odr() routine stopped." An empty list provides no information about why the routine stopped, failing to fulfill this contract.

2. **Silent failure in error reporting**: When used in `Output.pprint()` at line 619 of `_odrpack.py`, the code iterates over `self.stopreason` to display halt reasons:
   ```python
   print('Reason(s) for Halting:')
   for r in self.stopreason:
       print(f'  {r}')
   ```
   An empty list causes no error messages to be printed, leaving users confused about why the ODR routine stopped.

3. **Incomplete error handling**: The function has multiple gaps:
   - For `I[0]` values 1, 2, and 3 (info values 10000, 20000, 30000), if all sub-flags are zero, no messages are appended
   - For `I[0]` values 7, 8, and 9 (info values 70000-99999), there are no handlers at all
   - The function successfully handles `I[0]` values 0, 4, 5, and 6 but fails silently for others

## Relevant Context

The `_report_error` function is critical for user feedback in the ODR (Orthogonal Distance Regression) fitting process. The `info` parameter comes from the ODRPACK Fortran library and encodes various error and status conditions. The function decomposes `info` into five digit groups `I[0]` through `I[4]` where each digit represents different error categories.

Testing revealed that the bug affects six different `I[0]` values:
- `I[0] = 1`: Input dimension errors (when all sub-flags are 0)
- `I[0] = 2`: Array dimension errors (when all sub-flags are 0)
- `I[0] = 3`: Parameter errors (when all sub-flags are 0)
- `I[0] = 7`: Undefined error category
- `I[0] = 8`: Undefined error category
- `I[0] = 9`: Undefined error category

The ODRPACK User's Guide (referenced on page 38 according to the docstring) may define these error codes, but they are not handled in the current implementation.

## Proposed Fix

```diff
--- a/scipy/odr/_odrpack.py
+++ b/scipy/odr/_odrpack.py
@@ -150,6 +150,8 @@ def _report_error(info):
                 problems.append('NP < 1 or NP > N')
             if I[4] != 0:
                 problems.append('NQ < 1')
+            if not problems:  # No sub-flags set
+                problems.append('Input parameter dimension error')
         elif I[0] == 2:
             if I[1] != 0:
                 problems.append('LDY and/or LDX incorrect')
@@ -159,6 +161,8 @@ def _report_error(info):
                 problems.append('LDIFX, LDSTPD, and/or LDSCLD incorrect')
             if I[4] != 0:
                 problems.append('LWORK and/or LIWORK too small')
+            if not problems:  # No sub-flags set
+                problems.append('Array dimension error')
         elif I[0] == 3:
             if I[1] != 0:
                 problems.append('STPB and/or STPD incorrect')
@@ -168,6 +172,8 @@ def _report_error(info):
                 problems.append('WE incorrect')
             if I[4] != 0:
                 problems.append('WD incorrect')
+            if not problems:  # No sub-flags set
+                problems.append('Parameter specification error')
         elif I[0] == 4:
             problems.append('Error in derivatives')
         elif I[0] == 5:
             problems.append('Error occurred in callback')
         elif I[0] == 6:
             problems.append('Numerical error detected')
+        else:  # I[0] in (7, 8, 9) or any unexpected value
+            problems.append(f'Unknown error (info={info})')

         return problems
```