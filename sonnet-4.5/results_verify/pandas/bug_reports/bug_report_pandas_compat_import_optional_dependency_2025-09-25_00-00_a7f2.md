# Bug Report: pandas.compat import_optional_dependency violates errors parameter contract

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: High
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `import_optional_dependency` function raises `ImportError` when `errors='ignore'` or `errors='warn'`, violating its documented contract that these modes should handle errors gracefully without raising exceptions.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pandas.compat._optional import import_optional_dependency
import pytest


builtin_modules = st.sampled_from(['sys', 'os', 'io', 'math', 'json', 're', 'time'])


@given(builtin_modules, st.text(min_size=1, max_size=10))
def test_errors_ignore_never_raises_on_version_check(module_name, min_version):
    result = import_optional_dependency(module_name, min_version=min_version, errors='ignore')


@given(builtin_modules, st.text(min_size=1, max_size=10))
def test_errors_warn_never_raises_on_version_check(module_name, min_version):
    with pytest.warns():
        result = import_optional_dependency(module_name, min_version=min_version, errors='warn')
```

**Failing input**: `module_name='sys'`, `min_version='1.0.0'`, `errors='ignore'`

## Reproducing the Bug

```python
from pandas.compat._optional import import_optional_dependency

result = import_optional_dependency('sys', min_version='1.0.0', errors='ignore')
```

Output:
```
ImportError: Can't determine version for sys
```

## Why This Is A Bug

The function's docstring states:

> errors : str {'raise', 'warn', 'ignore'}
>     What to do when a dependency is not found or its version is too old.
>     * ignore: If the module is not installed, return None, otherwise,
>       return the module, even if the version is too old.

When `errors='ignore'`, the function should return the module even if version checking fails. Similarly, when `errors='warn'`, it should warn and return None, not raise. The bug occurs because `get_version()` raises `ImportError` before the `errors` parameter is checked.

## Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -147,7 +147,15 @@ def import_optional_dependency(
         module_to_get = module
     minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
     if minimum_version:
-        version = get_version(module_to_get)
+        try:
+            version = get_version(module_to_get)
+        except ImportError:
+            if errors == "warn":
+                warnings.warn(
+                    f"Can't determine version for {module_to_get.__name__}",
+                    UserWarning,
+                    stacklevel=find_stack_level(),
+                )
+                return None
+            elif errors == "ignore":
+                return module
+            else:
+                raise
         if version and Version(version) < Version(minimum_version):
             msg = (
                 f"Pandas requires version '{minimum_version}' or newer of '{parent}' "
```