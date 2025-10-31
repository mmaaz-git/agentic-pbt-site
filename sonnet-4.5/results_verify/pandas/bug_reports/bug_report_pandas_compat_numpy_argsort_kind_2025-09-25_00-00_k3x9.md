# Bug Report: pandas.compat.numpy ARGSORT_DEFAULTS Duplicate Assignment

**Target**: `pandas.compat.numpy.function.ARGSORT_DEFAULTS`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ARGSORT_DEFAULTS["kind"]` is assigned twice in consecutive lines, overwriting the correct numpy default value `"quicksort"` with `None`, causing validation to incorrectly reject the numpy default value.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

import pytest
from hypothesis import given, strategies as st
from pandas.compat.numpy import function as nv


@given(st.sampled_from(['quicksort', 'mergesort', 'heapsort', 'stable']))
def test_validate_argsort_with_kind_parameter(kind):
    validator = nv.validate_argsort
    validator((), {"kind": kind})
```

**Failing input**: `kind='quicksort'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages')

from pandas.compat.numpy.function import validate_argsort

validate_argsort((), {"kind": "quicksort"})
```

Expected: Validation passes (since `"quicksort"` is numpy's default for argsort)
Actual: Raises `ValueError: the 'kind' parameter is not supported in the pandas implementation of argsort()`

## Why This Is A Bug

The purpose of `ARGSORT_DEFAULTS` is to store the default values from numpy's `argsort` function, so that pandas can validate whether users are passing non-default values. Lines 139-140 show:

```python
ARGSORT_DEFAULTS["kind"] = "quicksort"  # Line 139: correct numpy default
ARGSORT_DEFAULTS["order"] = None
ARGSORT_DEFAULTS["kind"] = None  # Line 140: duplicate assignment overwrites line 139
```

This causes `validate_argsort` to reject `kind="quicksort"` even though it's numpy's default value, violating the stated purpose of numpy compatibility.

## Fix

```diff
--- a/pandas/compat/numpy/function.py
+++ b/pandas/compat/numpy/function.py
@@ -136,8 +136,7 @@ def validate_argmax_with_skipna(skipna: bool | ndarray | None, args, kwargs) -
 ARGSORT_DEFAULTS: dict[str, int | str | None] = {}
 ARGSORT_DEFAULTS["axis"] = -1
 ARGSORT_DEFAULTS["kind"] = "quicksort"
 ARGSORT_DEFAULTS["order"] = None
-ARGSORT_DEFAULTS["kind"] = None
 ARGSORT_DEFAULTS["stable"] = None


```