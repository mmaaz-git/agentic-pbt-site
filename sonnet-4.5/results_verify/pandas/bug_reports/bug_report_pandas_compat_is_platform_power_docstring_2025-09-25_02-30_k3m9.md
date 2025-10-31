# Bug Report: pandas.compat.is_platform_power Docstring Mismatch

**Target**: `pandas.compat.is_platform_power`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_platform_power()` function has incorrect documentation that states it returns "True if the running platform uses ARM architecture" when it actually checks for Power architecture.

## Property-Based Test

```python
def test_is_platform_power_docstring_accuracy():
    import inspect
    from pandas.compat import is_platform_power

    doc = is_platform_power.__doc__
    source = inspect.getsource(is_platform_power)

    assert "Power" in doc or "power" in doc or "ppc" in doc
```

**Failing input**: N/A (documentation bug)

## Reproducing the Bug

```python
import inspect
from pandas.compat import is_platform_power

doc = is_platform_power.__doc__
source = inspect.getsource(is_platform_power)

print("Docstring says:")
print(doc)
print("\nImplementation checks:")
print('platform.machine() in ("ppc64", "ppc64le")' in source)
print("\nThe docstring incorrectly states 'ARM architecture' instead of 'Power architecture'")
```

## Why This Is A Bug

The function checks for Power architecture (`ppc64`, `ppc64le`) but the Returns section of the docstring says "True if the running platform uses ARM architecture". This is a clear copy-paste error from the `is_platform_arm()` function defined just above it, which has identical docstring wording but checks for ARM instead.

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