# Bug Report: pandas.core.strings cat_core Empty List Type Error

**Target**: `pandas.core.strings.accessor.cat_core` and `pandas.core.strings.accessor.cat_safe`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`cat_core` and `cat_safe` return an integer `0` instead of a numpy array when given an empty list, violating their documented return type contract.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from hypothesis import given, strategies as st
import pandas.core.strings.accessor as accessor


@given(st.text())
def test_cat_core_empty_list_returns_array(sep):
    result = accessor.cat_core([], sep)
    assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result).__name__}: {result}"


@given(st.text())
def test_cat_safe_empty_list_returns_array(sep):
    result = accessor.cat_safe([], sep)
    assert isinstance(result, np.ndarray), f"Expected np.ndarray, got {type(result).__name__}: {result}"
```

**Failing input**: `cat_core([], sep)` for any `sep` value (e.g., `''`, `','`, etc.)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

import numpy as np
from pandas.core.strings.accessor import cat_core, cat_safe

result_core = cat_core([], ',')
print(f'cat_core([], ","): {repr(result_core)}')
print(f'Type: {type(result_core).__name__}')
assert isinstance(result_core, np.ndarray), "Expected np.ndarray, got int"

result_safe = cat_safe([], ',')
print(f'cat_safe([], ","): {repr(result_safe)}')
print(f'Type: {type(result_safe).__name__}')
assert isinstance(result_safe, np.ndarray), "Expected np.ndarray, got int"
```

## Why This Is A Bug

Both functions document in their docstrings that they return `nd.array`, but when given an empty list, they return the integer `0`. This happens because `np.sum([])` returns `0` by default. This violates the return type contract and could cause downstream code expecting an array to fail with AttributeError when trying to call array methods on the integer result.

## Fix

```diff
--- a/pandas/core/strings/accessor.py
+++ b/pandas/core/strings/accessor.py
@@ -14,6 +14,9 @@ def cat_core(list_of_columns: list, sep: str):
     nd.array
         The concatenation of list_of_columns with sep.
     """
+    if len(list_of_columns) == 0:
+        return np.array([], dtype=object)
+
     if sep == "":
         # no need to interleave sep if it is empty
         arr_of_cols = np.asarray(list_of_columns, dtype=object)
```