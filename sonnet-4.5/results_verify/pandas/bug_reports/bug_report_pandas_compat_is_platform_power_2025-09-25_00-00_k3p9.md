# Bug Report: pandas.compat.is_platform_power Documentation Error

**Target**: `pandas.compat.is_platform_power`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_platform_power()` function has incorrect documentation in its Returns section, claiming it returns "True if the running platform uses ARM architecture" when it actually checks for Power architecture.

## Property-Based Test

```python
def test_is_platform_power_docstring():
    import inspect
    import pandas.compat

    source = inspect.getsource(pandas.compat.is_platform_power)
    assert "ARM architecture" not in source or "Power architecture" in source.split("Returns")[1].split("---")[1]
```

**Failing observation**: The docstring's Returns section incorrectly states "ARM architecture" instead of "Power architecture".

## Reproducing the Bug

```python
import pandas.compat
import inspect

source = inspect.getsource(pandas.compat.is_platform_power)
print(source)
```

The function checks `platform.machine() in ("ppc64", "ppc64le")` which are Power architectures (PowerPC 64-bit), but the Returns section of the docstring incorrectly says "True if the running platform uses ARM architecture".

## Why This Is A Bug

The docstring contradicts the implementation. The function:
1. Is named `is_platform_power` (indicating Power architecture)
2. Has summary "Checking if the running platform use Power architecture"
3. Checks for "ppc64" and "ppc64le" (Power architecture machine types)

But the Returns section incorrectly says "ARM architecture", which is a copy-paste error from the `is_platform_arm()` function.

## Fix

```diff
--- a/pandas/compat/__init__.py
+++ b/pandas/compat/__init__.py
@@ -XX,7 +XX,7 @@ def is_platform_power() -> bool:
     Returns
     -------
     bool
-        True if the running platform uses ARM architecture.
+        True if the running platform uses Power architecture.
     """
     return platform.machine() in ("ppc64", "ppc64le")
```