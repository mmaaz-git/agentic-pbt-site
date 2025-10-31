# Bug Report: pandas.compat.is_platform_power Documentation Incorrect

**Target**: `pandas.compat.is_platform_power`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The docstring for `is_platform_power()` incorrectly states it returns "True if the running platform uses ARM architecture", but the implementation actually checks for Power architecture (PowerPC: ppc64, ppc64le).

## Property-Based Test

```python
def test_is_platform_power_documentation_matches_implementation():
    result = pandas.compat.is_platform_power()

    doc = pandas.compat.is_platform_power.__doc__
    assert "Power architecture" in doc

    if "ARM architecture" in doc:
        raise AssertionError(
            "Documentation for is_platform_power() incorrectly states it checks for "
            "ARM architecture, but the implementation checks for Power architecture (ppc64, ppc64le). "
            "This is a documentation bug."
        )
```

**Failing input**: N/A (documentation bug)

## Reproducing the Bug

```python
import pandas.compat
import inspect

print("Documentation says:")
print(pandas.compat.is_platform_power.__doc__)

print("\nImplementation:")
print(inspect.getsource(pandas.compat.is_platform_power))
```

Output shows:
```
Returns
-------
bool
    True if the running platform uses ARM architecture.
```

But implementation checks: `platform.machine() in ("ppc64", "ppc64le")`

PowerPC (PPC) is not ARM - these are completely different architectures.

## Why This Is A Bug

The docstring contradicts the implementation. The function is named `is_platform_power`, correctly checks for PowerPC machines (ppc64/ppc64le), but the return value documentation incorrectly says "ARM architecture" instead of "Power architecture". This would mislead users about what the function actually does.

## Fix

```diff
--- a/pandas/compat/__init__.py
+++ b/pandas/compat/__init__.py
@@ -95,7 +95,7 @@ def is_platform_power() -> bool:
     Returns
     -------
     bool
-        True if the running platform uses ARM architecture.
+        True if the running platform uses Power architecture.
     """
     return platform.machine() in ("ppc64", "ppc64le")
```