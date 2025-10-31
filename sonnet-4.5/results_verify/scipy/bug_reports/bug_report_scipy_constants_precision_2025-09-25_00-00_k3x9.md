# Bug Report: scipy.constants.precision() Docstring Example

**Target**: `scipy.constants.precision()`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring example for `scipy.constants.precision()` shows incorrect output that doesn't match the actual function behavior.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import scipy.constants

@given(st.sampled_from(list(scipy.constants.physical_constants.keys())))
def test_precision_returns_relative_not_absolute(key):
    val, unit, uncertainty = scipy.constants.physical_constants[key]
    result = scipy.constants.precision(key)

    if val != 0 and uncertainty != 0:
        expected_relative = uncertainty / val
        assert result == expected_relative, f"Expected {expected_relative}, got {result}"
```

**Failing input**: `key='proton mass'` (from the docstring example itself)

## Reproducing the Bug

```python
import scipy.constants

key = 'proton mass'
val, unit, uncertainty = scipy.constants.physical_constants[key]

result = scipy.constants.precision(key)

print(f"Documentation claims: 5.1e-37")
print(f"Actual result: {result}")
print(f"Match: {result == 5.1e-37}")
```

Output:
```
Documentation claims: 5.1e-37
Actual result: 3.1088914472088803e-10
Match: False
```

## Why This Is A Bug

The docstring for `scipy.constants.precision()` contains this example:

```python
>>> from scipy import constants
>>> constants.precision('proton mass')
5.1e-37
```

However, the actual output is `3.1088914472088803e-10` (approximately `3.11e-10`).

The function correctly computes **relative precision** (uncertainty / value), but the example shows a value close to the absolute uncertainty instead. This violates the API contract by providing misleading documentation.

## Fix

```diff
--- a/scipy/constants/_codata.py
+++ b/scipy/constants/_codata.py
@@ -123,7 +123,7 @@ def precision(key: str) -> float:
     Examples
     --------
     >>> from scipy import constants
     >>> constants.precision('proton mass')
-    5.1e-37
+    3.11e-10
```