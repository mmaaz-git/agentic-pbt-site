# Bug Report: pandas.compat.numpy.function ARGSORT_DEFAULTS Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary in `pandas.compat.numpy.function` contains a duplicate assignment to the `"kind"` key, where it is first set to `"quicksort"` and then immediately overwritten with `None`, causing incorrect default validation behavior.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS


@given(st.data())
def test_argsort_defaults_kind_should_not_be_duplicated(data):
    assert "kind" in ARGSORT_DEFAULTS
    kind_value = ARGSORT_DEFAULTS["kind"]

    assert kind_value is not None, (
        f"ARGSORT_DEFAULTS['kind'] should have a default value, "
        f"but got None. This appears to be due to duplicate assignment "
        f"where 'kind' is first set to 'quicksort' then overwritten to None."
    )
```

**Failing input**: Any input (the bug is in the constant initialization)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print("ARGSORT_DEFAULTS contents:")
for key, value in ARGSORT_DEFAULTS.items():
    print(f"  {key}: {value!r}")

print(f"\nBug: 'kind' is {ARGSORT_DEFAULTS['kind']!r}, expected 'quicksort'")
```

## Why This Is A Bug

In the source file at `/pandas/compat/numpy/function.py` lines 136-141:

```python
ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None  # Line 140 - duplicate assignment!
ARGSORT_DEFAULTS["stable"] = None
```

The key `"kind"` is assigned twice: first to `"quicksort"` (line 138) and then to `None` (line 140). The second assignment overwrites the first, resulting in `ARGSORT_DEFAULTS["kind"] = None` instead of the intended `"quicksort"`.

This causes the `validate_argsort` validator to use incorrect default values when validating parameters, potentially accepting invalid arguments or rejecting valid ones.

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -136,8 +136,7 @@ def validate_take_with_convert(convert: ndarray | bool | None, args, kwargs) ->
 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
 ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None


```