# Bug Report: pandas.compat.numpy.function ARGSORT_DEFAULTS Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The ARGSORT_DEFAULTS dictionary in pandas/compat/numpy/function.py contains a duplicate assignment of the "kind" key on lines 138 and 140, causing the second assignment (None) to overwrite the intended default value ("quicksort"). This causes the validator to incorrectly reject valid numpy-compatible parameters.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for ARGSORT_DEFAULTS duplicate key bug using Hypothesis."""

from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

def test_argsort_defaults_no_duplicate_keys():
    """Test that ARGSORT_DEFAULTS has the correct default value for 'kind'."""
    assert ARGSORT_DEFAULTS["kind"] == "quicksort", \
        f"Expected kind='quicksort', got kind={ARGSORT_DEFAULTS['kind']!r}"

if __name__ == "__main__":
    # Run the test
    try:
        test_argsort_defaults_no_duplicate_keys()
        print("Test passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
```

<details>

<summary>
**Failing input**: Not applicable (static configuration bug)
</summary>
```
Test failed: Expected kind='quicksort', got kind=None
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of the ARGSORT_DEFAULTS duplicate key bug."""

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS dictionary contents:")
print(ARGSORT_DEFAULTS)

print("\nChecking 'kind' parameter value:")
print(f"Actual 'kind' value: {ARGSORT_DEFAULTS['kind']!r}")
print(f"Expected 'kind' value: 'quicksort'")

print("\nAnalysis:")
if ARGSORT_DEFAULTS["kind"] is None:
    print("BUG CONFIRMED: The 'kind' key is None instead of 'quicksort'")
    print("This occurs because line 140 overwrites the value from line 138")
else:
    print("Bug not present - kind has the expected value")
```

<details>

<summary>
Output showing duplicate key overwrite
</summary>
```
ARGSORT_DEFAULTS dictionary contents:
{'axis': -1, 'kind': None, 'order': None, 'stable': None}

Checking 'kind' parameter value:
Actual 'kind' value: None
Expected 'kind' value: 'quicksort'

Analysis:
BUG CONFIRMED: The 'kind' key is None instead of 'quicksort'
This occurs because line 140 overwrites the value from line 138
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple ways:

1. **Code Error**: Lines 136-141 in pandas/compat/numpy/function.py contain an obvious duplicate dictionary key assignment:
   ```python
   ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138
   ARGSORT_DEFAULTS["order"] = None        # Line 139
   ARGSORT_DEFAULTS["kind"] = None         # Line 140 - overwrites line 138!
   ```

2. **NumPy Compatibility Violation**: NumPy's argsort documentation explicitly states "The default is 'quicksort'" even though the function signature shows `kind=None`. When None is passed, NumPy internally uses 'quicksort'. The validator should accept 'quicksort' as a valid default.

3. **Validation Failure**: The bug causes validate_argsort to incorrectly reject `kind="quicksort"`:
   ```
   validate_argsort((), {"kind": "quicksort"})
   # Raises: the 'kind' parameter is not supported in the pandas implementation of argsort()
   ```
   This error message is misleading - the parameter IS supported, but the validator has the wrong default stored.

4. **Intent Mismatch**: The first assignment (`kind="quicksort"`) clearly shows the developer's intent to match NumPy's default behavior. The duplicate assignment appears to be a copy-paste error or merge conflict artifact.

## Relevant Context

The pandas numpy compatibility layer exists to ensure pandas functions can accept numpy arguments for compatibility. According to the module docstring, the validators ensure users "only pass in the default values for these parameters" to discourage reliance on pandas-specific parameter handling.

There are actually two argsort validators defined:
- `validate_argsort` using `ARGSORT_DEFAULTS` (has the bug)
- `validate_argsort_kind` using `ARGSORT_DEFAULTS_KIND` (intentionally excludes 'kind' parameter entirely)

The comment on lines 148-149 mentions "two different signatures of argsort," suggesting confusion during development about how to handle the 'kind' parameter, with line 140 likely being a leftover that should have been removed.

NumPy argsort documentation: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html

## Proposed Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -136,8 +136,7 @@ validate_argmax = CompatValidator(
 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
 ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```