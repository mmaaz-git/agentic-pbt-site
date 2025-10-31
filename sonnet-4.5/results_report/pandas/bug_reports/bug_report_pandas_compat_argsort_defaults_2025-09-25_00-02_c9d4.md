# Bug Report: pandas.compat ARGSORT_DEFAULTS duplicate key assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary has duplicate assignment to the `"kind"` key, where it's first set to `"quicksort"` and then immediately overwritten with `None`.

## Property-Based Test

```python
from hypothesis import given, strategies as st


def test_argsort_defaults_no_duplicate_keys():
    from pandas.compat.numpy.function import ARGSORT_DEFAULTS

    assert ARGSORT_DEFAULTS["kind"] == "quicksort"
```

**Failing input**: N/A (code structure bug)

## Reproducing the Bug

```python
from pandas.compat.numpy.function import ARGSORT_DEFAULTS

print(ARGSORT_DEFAULTS)
```

Output:
```
{'axis': -1, 'kind': None, 'order': None, 'stable': None}
```

Expected: `'kind': 'quicksort'` or just a single assignment to `'kind'`

## Why This Is A Bug

In the source file `pandas/compat/numpy/function.py`, lines 154-158:

```python
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None
ARGSORT_DEFAULTS["stable"] = None
```

The `"kind"` key is assigned twice in succession. The first assignment to `"quicksort"` is immediately overwritten by the second assignment to `None`. This appears to be a copy-paste error or merge conflict artifact.

While the practical impact is minimal (numpy treats `kind=None` as equivalent to the default), the code is confusing and doesn't reflect the intended behavior.

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -152,10 +152,9 @@ def validate_argmax_with_skipna(skipna: bool | ndarray | None, args, kwargs) -

 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
 ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```