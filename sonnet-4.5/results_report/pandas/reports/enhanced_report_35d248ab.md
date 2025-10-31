# Bug Report: pandas.compat.numpy.function ARGSORT_DEFAULTS Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary in `pandas/compat/numpy/function.py` contains a duplicate assignment for the `"kind"` key, where line 138 sets it to `"quicksort"` and line 140 immediately overwrites it with `None`, making line 138 dead code.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for ARGSORT_DEFAULTS duplicate assignment bug.
"""

from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS


@given(st.just(None))
def test_argsort_defaults_duplicate_assignment(expected):
    """
    Test that ARGSORT_DEFAULTS['kind'] has the value None.

    This test reveals that line 138 in function.py which sets
    'kind' to 'quicksort' is dead code, as it's immediately
    overwritten by line 140 which sets it to None.
    """
    actual = ARGSORT_DEFAULTS['kind']
    assert actual == expected, f"Expected {expected} but got {actual}"

    # Additional assertion to demonstrate the dead code
    # Line 138 sets 'quicksort', but it never takes effect
    assert actual != 'quicksort', "Line 138's assignment should be overwritten"


if __name__ == "__main__":
    # Run the hypothesis test
    test_argsort_defaults_duplicate_assignment()
    print("Hypothesis test passed: ARGSORT_DEFAULTS['kind'] is None")
    print("This confirms that line 138 (setting 'kind' to 'quicksort') is dead code")
```

<details>

<summary>
**Failing input**: `None` (fixed test value)
</summary>
```
Hypothesis test passed: ARGSORT_DEFAULTS['kind'] is None
This confirms that line 138 (setting 'kind' to 'quicksort') is dead code
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Reproduction script for ARGSORT_DEFAULTS duplicate assignment bug.
"""

# Import the ARGSORT_DEFAULTS dictionary
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("=== ARGSORT_DEFAULTS Duplicate Assignment Bug ===")
print()

# Show the current value of ARGSORT_DEFAULTS
print("Current ARGSORT_DEFAULTS dictionary:")
print(f"  ARGSORT_DEFAULTS = {ARGSORT_DEFAULTS}")
print()

# Show the specific 'kind' value
print("Value of ARGSORT_DEFAULTS['kind']:")
print(f"  ARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']}")
print()

# Demonstrate the issue
print("Issue in pandas/compat/numpy/function.py lines 138-140:")
print("  Line 138: ARGSORT_DEFAULTS['kind'] = 'quicksort'")
print("  Line 139: ARGSORT_DEFAULTS['order'] = None")
print("  Line 140: ARGSORT_DEFAULTS['kind'] = None  # <-- Overwrites line 138!")
print()

# Show what the expected behavior would be
print("Analysis:")
print("  - Line 138 sets 'kind' to 'quicksort'")
print("  - Line 140 immediately overwrites it with None")
print("  - Result: Line 138 is dead code with no effect")
print()

# Verify the final value
expected_from_line_138 = "quicksort"
actual_value = ARGSORT_DEFAULTS['kind']

print("Verification:")
print(f"  Expected from line 138: '{expected_from_line_138}'")
print(f"  Actual final value: {actual_value}")
print(f"  Match: {actual_value == expected_from_line_138}")
print()

if actual_value != expected_from_line_138:
    print("CONFIRMED: Line 138 is dead code - its assignment is immediately overwritten")
else:
    print("ERROR: Unexpected state - line 140 should have overwritten line 138")
```

<details>

<summary>
Output demonstrating dead code issue
</summary>
```
=== ARGSORT_DEFAULTS Duplicate Assignment Bug ===

Current ARGSORT_DEFAULTS dictionary:
  ARGSORT_DEFAULTS = {'axis': -1, 'kind': None, 'order': None, 'stable': None}

Value of ARGSORT_DEFAULTS['kind']:
  ARGSORT_DEFAULTS['kind'] = None

Issue in pandas/compat/numpy/function.py lines 138-140:
  Line 138: ARGSORT_DEFAULTS['kind'] = 'quicksort'
  Line 139: ARGSORT_DEFAULTS['order'] = None
  Line 140: ARGSORT_DEFAULTS['kind'] = None  # <-- Overwrites line 138!

Analysis:
  - Line 138 sets 'kind' to 'quicksort'
  - Line 140 immediately overwrites it with None
  - Result: Line 138 is dead code with no effect

Verification:
  Expected from line 138: 'quicksort'
  Actual final value: None
  Match: False

CONFIRMED: Line 138 is dead code - its assignment is immediately overwritten
```
</details>

## Why This Is A Bug

This is a clear case of dead code in the pandas source. The code at lines 136-141 in `pandas/compat/numpy/function.py` contains:

```python
ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138 - sets kind to "quicksort"
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None         # Line 140 - immediately overwrites with None
ARGSORT_DEFAULTS["stable"] = None
```

The assignment `ARGSORT_DEFAULTS["kind"] = "quicksort"` at line 138 has no effect because it is immediately overwritten two lines later by `ARGSORT_DEFAULTS["kind"] = None` at line 140. This violates basic code quality principles:

1. **Dead code**: Line 138 is unreachable in its effect - it executes but its result is never used
2. **Confusing intent**: Having two assignments to the same key suggests either a copy-paste error or leftover code from refactoring
3. **Maintenance burden**: Future maintainers may be confused about which value is intended

While this doesn't cause runtime errors (the final value `None` matches NumPy 2.x defaults), it represents poor code quality and should be cleaned up.

## Relevant Context

The module `pandas/compat/numpy/function.py` provides default arguments for numpy compatibility. According to the module docstring, these defaults are used to validate that users only pass default values for numpy parameters that pandas doesn't actually use.

Looking at the code structure:
- `ARGSORT_DEFAULTS` is used for validating argsort function calls (line 144-146)
- There's also `ARGSORT_DEFAULTS_KIND` (lines 150-156) which notably does NOT include a "kind" key at all
- The comment on lines 148-149 suggests `ARGSORT_DEFAULTS_KIND` is "for when the 'kind' param is supported", yet it doesn't contain 'kind'

NumPy's current `argsort` function signature has `kind=None` as the default (verified with NumPy 2.3.0), so the final value of `None` appears correct. The `"quicksort"` value on line 138 may be leftover from older NumPy versions where quicksort was the default.

Code location: [pandas/compat/numpy/function.py lines 136-141](https://github.com/pandas-dev/pandas/blob/main/pandas/compat/numpy/function.py#L136-L141)

## Proposed Fix

Remove the dead code assignment at line 138:

```diff
 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
 ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```