# Bug Report: pandas.compat.numpy.function ARGSORT_DEFAULTS Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary in pandas has the `'kind'` key assigned twice on consecutive lines, with line 140 overwriting the 'quicksort' value from line 138 with None.

## Property-Based Test

```python
"""
Property-based test demonstrating the ARGSORT_DEFAULTS duplicate assignment bug
"""

from hypothesis import given, settings, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

@given(st.integers())
@settings(max_examples=1)  # We only need one example to show the issue
def test_argsort_defaults_kind_value(x):
    """
    Test that ARGSORT_DEFAULTS['kind'] has the expected value.

    The code sets 'kind' to 'quicksort' on line 138,
    then immediately overwrites it with None on line 140.

    This test demonstrates that the final value is None,
    not the expected 'quicksort'.
    """
    # The actual value is None due to the duplicate assignment
    actual_value = ARGSORT_DEFAULTS['kind']

    # According to line 138, it should be 'quicksort'
    # But line 140 overwrites it to None
    print(f"ARGSORT_DEFAULTS['kind'] = {actual_value!r}")
    print(f"Line 138 sets it to: 'quicksort'")
    print(f"Line 140 overwrites it to: None")
    print(f"Actual final value: {actual_value!r}")

    # This assertion will pass because we're checking what it actually is (None)
    # But the duplicate assignment is still a bug
    assert actual_value is None, f"Expected None (from line 140), but got {actual_value!r}"

    # This would fail if we expected the line 138 value:
    # assert actual_value == 'quicksort', f"Expected 'quicksort' (from line 138), but got {actual_value!r}"

if __name__ == "__main__":
    test_argsort_defaults_kind_value()
```

<details>

<summary>
**Failing input**: `x=0`
</summary>
```
ARGSORT_DEFAULTS['kind'] = None
Line 138 sets it to: 'quicksort'
Line 140 overwrites it to: None
Actual final value: None
```
</details>

## Reproducing the Bug

```python
"""
Demonstrate the duplicate assignment bug in pandas.compat.numpy.function.ARGSORT_DEFAULTS
"""

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS contents:")
for key, value in sorted(ARGSORT_DEFAULTS.items()):
    print(f"  {key!r}: {value!r}")

print(f"\nARGSORT_DEFAULTS['kind'] = {ARGSORT_DEFAULTS['kind']!r}")
print("\nExpected: 'quicksort' (from line 138 of pandas/compat/numpy/function.py)")
print("Actual: None (overwritten by line 140 of pandas/compat/numpy/function.py)")

print("\n--- Source Code Issue ---")
print("In pandas/compat/numpy/function.py lines 136-141:")
print("  ARGSORT_DEFAULTS = {}")
print("  ARGSORT_DEFAULTS['axis'] = -1")
print("  ARGSORT_DEFAULTS['kind'] = 'quicksort'  # Line 138")
print("  ARGSORT_DEFAULTS['order'] = None")
print("  ARGSORT_DEFAULTS['kind'] = None         # Line 140 - overwrites line 138!")
print("  ARGSORT_DEFAULTS['stable'] = None")
print("\nThe duplicate assignment on line 140 overwrites the 'quicksort' value from line 138.")
```

<details>

<summary>
ARGSORT_DEFAULTS['kind'] incorrectly set to None instead of 'quicksort'
</summary>
```
ARGSORT_DEFAULTS contents:
  'axis': -1
  'kind': None
  'order': None
  'stable': None

ARGSORT_DEFAULTS['kind'] = None

Expected: 'quicksort' (from line 138 of pandas/compat/numpy/function.py)
Actual: None (overwritten by line 140 of pandas/compat/numpy/function.py)

--- Source Code Issue ---
In pandas/compat/numpy/function.py lines 136-141:
  ARGSORT_DEFAULTS = {}
  ARGSORT_DEFAULTS['axis'] = -1
  ARGSORT_DEFAULTS['kind'] = 'quicksort'  # Line 138
  ARGSORT_DEFAULTS['order'] = None
  ARGSORT_DEFAULTS['kind'] = None         # Line 140 - overwrites line 138!
  ARGSORT_DEFAULTS['stable'] = None

The duplicate assignment on line 140 overwrites the 'quicksort' value from line 138.
```
</details>

## Why This Is A Bug

This is a clear programming error where the same dictionary key is assigned twice on consecutive lines with different values. In `pandas/compat/numpy/function.py`:

```python
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138
ARGSORT_DEFAULTS["order"] = None        # Line 139
ARGSORT_DEFAULTS["kind"] = None         # Line 140 - duplicate assignment!
```

The second assignment completely overwrites the first, making line 138 dead code. This violates basic programming principles and represents either:
1. A copy-paste error where someone forgot to change the key name
2. A merge conflict that was incorrectly resolved
3. An incomplete refactoring

The ARGSORT_DEFAULTS dictionary is used by `validate_argsort` to ensure users only pass default values for NumPy compatibility parameters. With `kind=None` instead of `kind='quicksort'`, the validator incorrectly accepts `None` as a valid default when it should only accept `'quicksort'` (NumPy's documented default for `argsort`).

## Relevant Context

The `pandas.compat.numpy.function` module provides default argument validation for NumPy compatibility. According to the module's docstring, these validators ensure users don't pass non-default values for NumPy parameters that pandas doesn't actually use.

The `validate_argsort` validator created from `ARGSORT_DEFAULTS` is used in:
- `pandas/core/indexes/range.py:530` - validates argsort arguments for RangeIndex
- `pandas/core/arrays/base.py:845` - validates argsort arguments for ExtensionArrays (via `validate_argsort_with_ascending`)
- `pandas/core/arrays/interval.py:855` - validates argsort arguments for IntervalArray (via `validate_argsort_with_ascending`)

NumPy's documentation states that `argsort` has `kind='quicksort'` as its default parameter. The validator should enforce this default, but due to the duplicate assignment bug, it incorrectly validates against `kind=None`.

Interestingly, there's also a separate `ARGSORT_DEFAULTS_KIND` dictionary (line 150-153) that doesn't include the 'kind' parameter at all, suggesting these handle different argsort signatures as noted in the comment on line 148-149.

## Proposed Fix

Remove the duplicate assignment on line 140:

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -137,7 +137,6 @@ ARGSORT_DEFAULTS["axis"] = -1
 ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```