# Bug Report: pandas.compat is_platform_power docstring mismatch

**Target**: `pandas.compat.is_platform_power`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `is_platform_power` function has a docstring that incorrectly states it returns "True if the running platform uses ARM architecture", when it actually checks for Power architecture (ppc64/ppc64le).

## Property-Based Test

```python
from hypothesis import given, strategies as st
import inspect


def test_platform_power_docstring_consistency():
    from pandas.compat import is_platform_power

    source = inspect.getsource(is_platform_power)
    doc = is_platform_power.__doc__

    assert "ppc64" in source
    assert "ARM" not in doc or "Power" in doc
```

**Failing input**: N/A (documentation bug)

## Reproducing the Bug

```python
from pandas.compat import is_platform_power
import inspect

print(inspect.getsource(is_platform_power))
```

Output shows docstring says "ARM architecture" but implementation checks for Power:
```python
def is_platform_power() -> bool:
    """
    Checking if the running platform use Power architecture.

    Returns
    -------
    bool
        True if the running platform uses ARM architecture.
    """
    return platform.machine() in ("ppc64", "ppc64le")
```

## Why This Is A Bug

The function summary correctly says "Checking if the running platform use Power architecture" but the Returns section incorrectly says "True if the running platform uses ARM architecture". This is a copy-paste error from `is_platform_arm` that creates confusion for developers reading the documentation.

## Fix

```diff
--- a/pandas/compat/__init__.py
+++ b/pandas/compat/__init__.py
@@ -111,7 +111,7 @@ def is_platform_power() -> bool:
     Returns
     -------
     bool
-        True if the running platform uses ARM architecture.
+        True if the running platform uses Power architecture.
     """
     return platform.machine() in ("ppc64", "ppc64le")
```