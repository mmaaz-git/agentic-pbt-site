# Bug Report: pandas.compat.numpy ARGSORT_DEFAULTS Duplicate Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary contains dead code where the `"kind"` key is assigned twice (lines 138 and 140), with the second assignment immediately overwriting the first. This results in `kind=None` being the default instead of `kind="quicksort"`, making the first assignment meaningless.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.compat.numpy.function import ARGSORT_DEFAULTS, validate_argsort


@given(st.sampled_from([None, "quicksort", "mergesort", "heapsort", "stable"]))
def test_validate_argsort_kind_consistency(kind_value):
    if kind_value == "quicksort":
        validate_argsort((), {"kind": kind_value})
```

**Failing input**: `kind_value = "quicksort"`

## Reproducing the Bug

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS, validate_argsort

print(ARGSORT_DEFAULTS)

validate_argsort((), {"kind": "quicksort"})
```

**Output**:
```
{'axis': -1, 'kind': None, 'order': None, 'stable': None}
ValueError: the 'kind' parameter is not supported in the pandas implementation of argsort()
```

## Why This Is A Bug

In `pandas/compat/numpy/function.py` lines 136-141:

```python
ARGSORT_DEFAULTS = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None         # Line 140 - overwrites!
ARGSORT_DEFAULTS["stable"] = None
```

The `"kind"` key is assigned twice. The first assignment to `"quicksort"` is dead code that is immediately overwritten by the second assignment to `None`. This means:

1. The default value for `kind` is `None`, not `"quicksort"` as line 138 suggests
2. Users cannot pass `kind="quicksort"` even though it's a valid numpy parameter
3. The first assignment serves no purpose and creates confusion

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -135,10 +135,8 @@ def validate_groupby_func(name: str, args, kwargs, allowed=None) -> None:

 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
 ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None


 validate_argsort = CompatValidator(
```

Remove line 138 to eliminate the dead code and make the intent clear that `kind=None` is the default.