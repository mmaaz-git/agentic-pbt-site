# Bug Report: pandas.compat.is_platform_power Incorrect Docstring

**Target**: `pandas.compat.is_platform_power`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring for `is_platform_power()` incorrectly states it returns "True if the running platform uses ARM architecture" when it actually checks for Power architecture.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
import pandas.compat

@given(st.integers())
@settings(max_examples=10)
def test_is_platform_power_docstring_matches_behavior(x):
    doc = pandas.compat.is_platform_power.__doc__
    assert "Power" in doc or "POWER" in doc, "Docstring should mention Power architecture"
    assert "ARM" not in doc or "arm" not in doc.lower(), "Docstring should not mention ARM"
```

**Failing input**: `x=0` (any value triggers the bug)

## Reproducing the Bug

```python
import pandas.compat

doc = pandas.compat.is_platform_power.__doc__
print(doc)
```

**Output**:
```
Checking if the running platform use Power architecture.

Returns
-------
bool
    True if the running platform uses ARM architecture.
```

The Returns section says "ARM architecture" but should say "Power architecture".

## Why This Is A Bug

The function implementation checks for Power/PowerPC architecture:
```python
return platform.machine() in ("ppc64", "ppc64le")
```

But the docstring's Returns section says "True if the running platform uses ARM architecture", which is copy-pasted from `is_platform_arm()` and not updated.

## Fix

```diff
--- a/pandas/compat/__init__.py
+++ b/pandas/compat/__init__.py
@@ -122,7 +122,7 @@ def is_platform_power() -> bool:
     Returns
     -------
     bool
-        True if the running platform uses ARM architecture.
+        True if the running platform uses Power architecture.
     """
     return platform.machine() in ("ppc64", "ppc64le")
```