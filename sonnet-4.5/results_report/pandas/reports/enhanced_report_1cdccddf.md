# Bug Report: pandas.compat.numpy.function ARGSORT_DEFAULTS Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The ARGSORT_DEFAULTS dictionary contains a duplicate assignment to the 'kind' key, where it is first set to 'quicksort' (line 138) and then immediately overwritten with None (line 140), causing the dictionary to have an incorrect default value.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for ARGSORT_DEFAULTS duplicate key bug"""

from hypothesis import given, strategies as st
import pytest


def test_argsort_defaults_no_duplicate_keys():
    """Test that ARGSORT_DEFAULTS['kind'] has the expected value of 'quicksort'.

    According to NumPy documentation, the default value for the 'kind' parameter
    in numpy.argsort is 'quicksort'. The pandas compatibility layer should
    reflect this default value.

    However, due to a duplicate assignment in the source code, the value is
    incorrectly set to None.
    """
    from pandas.compat.numpy.function import ARGSORT_DEFAULTS

    # This assertion will fail because of the duplicate assignment bug
    assert ARGSORT_DEFAULTS["kind"] == "quicksort", \
        f"Expected 'quicksort', got {ARGSORT_DEFAULTS['kind']}"


if __name__ == "__main__":
    # Run the test
    try:
        test_argsort_defaults_no_duplicate_keys()
        print("✓ Test passed: ARGSORT_DEFAULTS['kind'] == 'quicksort'")
    except AssertionError as e:
        print("✗ Test failed:")
        print(f"  {e}")
        print()
        print("Bug details:")
        from pandas.compat.numpy.function import ARGSORT_DEFAULTS
        print(f"  Current ARGSORT_DEFAULTS: {ARGSORT_DEFAULTS}")
        print("  The 'kind' key is set twice in the source code:")
        print("    Line 138: ARGSORT_DEFAULTS['kind'] = 'quicksort'")
        print("    Line 140: ARGSORT_DEFAULTS['kind'] = None  (overwrites previous)")
        import sys
        sys.exit(1)
```

<details>

<summary>
**Failing input**: N/A (static code structure bug, not input-dependent)
</summary>
```
✗ Test failed:
  Expected 'quicksort', got None

Bug details:
  Current ARGSORT_DEFAULTS: {'axis': -1, 'kind': None, 'order': None, 'stable': None}
  The 'kind' key is set twice in the source code:
    Line 138: ARGSORT_DEFAULTS['kind'] = 'quicksort'
    Line 140: ARGSORT_DEFAULTS['kind'] = None  (overwrites previous)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Demonstrate the duplicate assignment bug in pandas ARGSORT_DEFAULTS"""

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

# Show the current value
print("Current ARGSORT_DEFAULTS dictionary:")
print(ARGSORT_DEFAULTS)
print()

# Show that 'kind' is None, not 'quicksort'
print(f"Value of ARGSORT_DEFAULTS['kind']: {ARGSORT_DEFAULTS['kind']}")
print(f"Type of ARGSORT_DEFAULTS['kind']: {type(ARGSORT_DEFAULTS['kind'])}")
print()

# According to numpy documentation, the default for 'kind' should be 'quicksort'
print("Expected value based on NumPy documentation: 'quicksort'")
print(f"Actual value: {ARGSORT_DEFAULTS['kind']}")
print()

# Check if the value equals what we expect
if ARGSORT_DEFAULTS['kind'] == 'quicksort':
    print("✓ The 'kind' key has the expected value 'quicksort'")
else:
    print("✗ BUG: The 'kind' key is None instead of 'quicksort'")
    print("  This is due to duplicate assignment in the source code:")
    print("  Line 138: ARGSORT_DEFAULTS['kind'] = 'quicksort'")
    print("  Line 140: ARGSORT_DEFAULTS['kind'] = None  # Overwrites the previous value")
```

<details>

<summary>
BUG: The 'kind' key is None instead of 'quicksort' due to duplicate assignment
</summary>
```
Current ARGSORT_DEFAULTS dictionary:
{'axis': -1, 'kind': None, 'order': None, 'stable': None}

Value of ARGSORT_DEFAULTS['kind']: None
Type of ARGSORT_DEFAULTS['kind']: <class 'NoneType'>

Expected value based on NumPy documentation: 'quicksort'
Actual value: None

✗ BUG: The 'kind' key is None instead of 'quicksort'
  This is due to duplicate assignment in the source code:
  Line 138: ARGSORT_DEFAULTS['kind'] = 'quicksort'
  Line 140: ARGSORT_DEFAULTS['kind'] = None  # Overwrites the previous value
```
</details>

## Why This Is A Bug

This bug violates the expected behavior in multiple ways:

1. **Contradicts NumPy's documented defaults**: According to NumPy's official documentation for `numpy.argsort`, the 'kind' parameter has a default value of 'quicksort'. The pandas compatibility layer should accurately represent NumPy's defaults to ensure consistent behavior.

2. **Nonsensical duplicate assignment**: The code assigns 'quicksort' to ARGSORT_DEFAULTS['kind'] on line 138, then immediately overwrites it with None on line 140. This pattern serves no logical purpose and appears to be an unintentional error (likely a copy-paste mistake or merge conflict artifact).

3. **Inconsistent with related code**: The codebase also defines ARGSORT_DEFAULTS_KIND (lines 150-153) which notably does NOT include a 'kind' key at all, suggesting the duplicate assignment in ARGSORT_DEFAULTS is erroneous.

4. **Code clarity issue**: Even if NumPy internally treats kind=None as equivalent to kind='quicksort', the pandas code should clearly and accurately represent the intended defaults rather than containing confusing duplicate assignments.

## Relevant Context

The bug is located in `/pandas/compat/numpy/function.py` at lines 136-141:

```python
ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138: First assignment
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None         # Line 140: Second assignment (overwrites)
ARGSORT_DEFAULTS["stable"] = None
```

This dictionary is used to create a `CompatValidator` object called `validate_argsort` (line 144-146) which provides parameter validation for pandas' NumPy compatibility layer.

NumPy documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.argsort.html

The practical impact is minimal since NumPy likely treats kind=None as its default internally, but the code quality issue makes the codebase harder to understand and maintain.

## Proposed Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -135,11 +135,10 @@

 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
+ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None


 validate_argsort = CompatValidator(
     ARGSORT_DEFAULTS, fname="argsort", max_fname_arg_count=0, method="both"
```