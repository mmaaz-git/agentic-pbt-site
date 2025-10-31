# Bug Report: pandas.compat.import_optional_dependency KeyError for Submodules

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

When `import_optional_dependency` is called with a submodule name and `min_version` is specified, if the parent module is not in `sys.modules`, the function raises `KeyError` instead of the documented `ImportError`.

## Property-Based Test

```python
import sys
import pytest
from hypothesis import given, strategies as st, settings
from pandas.compat._optional import import_optional_dependency


@given(st.text(alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), min_codepoint=ord('a')), min_size=5))
@settings(max_examples=100)
def test_import_optional_dependency_submodule_no_keyerror(parent_name):
    parent_module_name = f"fake_parent_{parent_name}"
    submodule_name = f"{parent_module_name}.submodule"

    class FakeSubmodule:
        __name__ = submodule_name
        __version__ = "1.0.0"

    sys.modules[submodule_name] = FakeSubmodule()

    if parent_module_name in sys.modules:
        del sys.modules[parent_module_name]

    try:
        with pytest.raises((ImportError, KeyError)) as exc_info:
            import_optional_dependency(submodule_name, errors="raise", min_version="0.0.1")

        if isinstance(exc_info.value, KeyError):
            raise AssertionError(f"KeyError raised instead of ImportError: {exc_info.value}")
    finally:
        if submodule_name in sys.modules:
            del sys.modules[submodule_name]
```

**Failing input**: Any submodule name where the parent module is not in `sys.modules`

## Reproducing the Bug

```python
import sys
from pandas.compat._optional import import_optional_dependency


parent_module_name = "fake_parent_xyz"
submodule_name = f"{parent_module_name}.submodule"

class FakeSubmodule:
    __name__ = submodule_name
    __version__ = "1.0.0"

sys.modules[submodule_name] = FakeSubmodule()

if parent_module_name in sys.modules:
    del sys.modules[parent_module_name]

try:
    import_optional_dependency(submodule_name, errors="raise", min_version="0.0.1")
except KeyError as e:
    print(f"KeyError raised: {e}")
except ImportError as e:
    print(f"ImportError raised: {e}")
finally:
    del sys.modules[submodule_name]
```

## Why This Is A Bug

The function's docstring states it should raise `ImportError` on issues. However, when a submodule is imported with `min_version` specified, the code attempts to access `sys.modules[install_name]` without checking if it exists, causing a `KeyError` instead of the documented `ImportError`.

This violates the API contract and makes error handling unpredictable.

## Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -142,7 +142,11 @@ def import_optional_dependency(
     parent = name.split(".")[0]
     if parent != name:
         install_name = parent
-        module_to_get = sys.modules[install_name]
+        try:
+            module_to_get = sys.modules[install_name]
+        except KeyError:
+            raise ImportError(f"Parent module '{install_name}' not found in sys.modules")
     else:
         module_to_get = module
     minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
```