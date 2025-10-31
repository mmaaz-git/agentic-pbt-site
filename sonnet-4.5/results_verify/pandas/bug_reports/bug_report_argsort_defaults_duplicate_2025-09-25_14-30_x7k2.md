# Bug Report: pandas.compat.numpy.function ARGSORT_DEFAULTS Duplicate Key Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `ARGSORT_DEFAULTS` dictionary has a duplicate key assignment for "kind", where line 138 assigns `"quicksort"` and line 140 immediately overrides it with `None`. This appears to be a copy-paste error.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.compat.numpy.function as npfunc


def test_argsort_defaults_consistency():
    defaults = npfunc.ARGSORT_DEFAULTS
    assert defaults["kind"] is None
```

**Failing input**: N/A (this is a code organization issue)

## Reproducing the Bug

```python
import pandas.compat.numpy.function as npfunc

defaults = npfunc.ARGSORT_DEFAULTS

for key, value in defaults.items():
    print(f"{key}: {value!r}")

print(f"\nARGSORT_DEFAULTS['kind'] = {defaults['kind']!r}")
```

Output shows that `kind` is `None`, not `"quicksort"`, due to the duplicate assignment on line 140 overriding line 138.

## Why This Is A Bug

Looking at the source code in `pandas/compat/numpy/function.py` lines 136-141:

```python
ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
ARGSORT_DEFAULTS["axis"] = -1
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 138
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None         # Line 140 - overrides line 138!
ARGSORT_DEFAULTS["stable"] = None
```

The duplicate assignment suggests a copy-paste error. The comment on lines 148-149 explains that there are two validators:
- `ARGSORT_DEFAULTS` - for when the `kind` parameter is NOT supported
- `ARGSORT_DEFAULTS_KIND` - for when the `kind` parameter IS supported

Since `ARGSORT_DEFAULTS` is meant for cases where `kind` is not supported, it should have `kind=None`, making line 138's assignment of `"quicksort"` incorrect and unnecessary.

## Fix

Remove the duplicate assignment on line 138:

```diff
 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
-ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
 ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None
```