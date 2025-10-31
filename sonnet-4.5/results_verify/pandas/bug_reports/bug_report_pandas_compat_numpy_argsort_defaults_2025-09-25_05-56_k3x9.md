# Bug Report: pandas.compat.numpy ARGSORT_DEFAULTS Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary has the `"kind"` key assigned twice (lines 138 and 140), causing the second assignment to overwrite the first. This results in `kind` defaulting to `None` instead of `"quicksort"`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

def test_argsort_defaults_no_duplicate_assignments():
    """
    Property: Dictionary keys should only be assigned once during initialization.
    If 'kind' is meant to default to 'quicksort', it should not be overwritten.
    """
    assert ARGSORT_DEFAULTS["kind"] == "quicksort", \
        f"Expected 'quicksort' but got {ARGSORT_DEFAULTS['kind']}"
```

**Failing input**: N/A (static code analysis reveals the bug)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS:", ARGSORT_DEFAULTS)
print(f"kind value: {ARGSORT_DEFAULTS['kind']}")
print(f"Expected: 'quicksort'")
print(f"Actual: {ARGSORT_DEFAULTS['kind']}")

assert ARGSORT_DEFAULTS['kind'] is None, "Bug confirmed: 'kind' was overwritten to None"
```

## Why This Is A Bug

In `/pandas/compat/numpy/function.py` lines 136-141:

```python
ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None         # Line 140 - overwrites line 138!
ARGSORT_DEFAULTS["stable"] = None
```

Line 140 assigns `"kind"` to `None`, overwriting the `"quicksort"` value from line 138. This is clearly unintentional, as:

1. Duplicate key assignments in dictionary initialization are a common copy-paste error
2. NumPy's `argsort` accepts `kind` parameter with default values like `"quicksort"`
3. The presence of both assignments suggests one is a mistake

The validator will now use `None` as the default for `kind`, which may not match NumPy's actual default behavior.

## Fix

Remove the duplicate assignment on line 140:

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -136,8 +136,7 @@ ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
 ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
-ARGSORT_DEFAULTS["stable"] = None
+ARGSORT_DEFAULTS["stable"] = None

 validate_argsort = CompatValidator(
     ARGSORT_DEFAULTS, fname="argsort", max_fname_arg_count=0, method="both"
```