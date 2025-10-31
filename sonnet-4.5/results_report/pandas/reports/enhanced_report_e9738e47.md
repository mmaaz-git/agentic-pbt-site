# Bug Report: pandas.compat.import_optional_dependency Raises KeyError Instead of ImportError for Submodules

**Target**: `pandas.compat._optional.import_optional_dependency`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `import_optional_dependency` function raises an undocumented `KeyError` instead of the documented `ImportError` when importing a submodule with a `min_version` parameter if the parent module is not present in `sys.modules`.

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

# Run the test
if __name__ == "__main__":
    test_import_optional_dependency_submodule_no_keyerror()
```

<details>

<summary>
**Failing input**: `parent_name='aaaaa'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 34, in <module>
    test_import_optional_dependency_submodule_no_keyerror()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 8, in test_import_optional_dependency_submodule_no_keyerror
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/40/hypo.py", line 27, in test_import_optional_dependency_submodule_no_keyerror
    raise AssertionError(f"KeyError raised instead of ImportError: {exc_info.value}")
AssertionError: KeyError raised instead of ImportError: 'fake_parent_aaaaa'
Falsifying example: test_import_optional_dependency_submodule_no_keyerror(
    parent_name='aaaaa',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
from pandas.compat._optional import import_optional_dependency

# Set up the test scenario
parent_module_name = "fake_parent_xyz"
submodule_name = f"{parent_module_name}.submodule"

# Create a fake submodule with a version
class FakeSubmodule:
    __name__ = submodule_name
    __version__ = "1.0.0"

# Add the submodule to sys.modules
sys.modules[submodule_name] = FakeSubmodule()

# Ensure the parent module is NOT in sys.modules
if parent_module_name in sys.modules:
    del sys.modules[parent_module_name]

# Try to import the submodule with a min_version requirement
try:
    result = import_optional_dependency(submodule_name, errors="raise", min_version="0.0.1")
    print(f"Successfully imported: {result}")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print(f"Exception type: {type(e).__name__}")
except ImportError as e:
    print(f"ImportError raised: {e}")
    print(f"Exception type: {type(e).__name__}")
finally:
    # Clean up sys.modules
    if submodule_name in sys.modules:
        del sys.modules[submodule_name]
```

<details>

<summary>
KeyError raised with wrong exception type
</summary>
```
KeyError raised: 'fake_parent_xyz'
Exception type: KeyError
```
</details>

## Why This Is A Bug

This violates the documented API contract of `import_optional_dependency`. The function's docstring explicitly states on lines 94-95 and 107 that "By default, if a dependency is missing an ImportError with a nice message will be raised" and when `errors='raise'`: "Raise an ImportError".

The bug occurs at line 145 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/compat/_optional.py` where the code executes `module_to_get = sys.modules[install_name]` without checking if the parent module exists in `sys.modules`. This happens specifically when:

1. A submodule (e.g., `parent.child`) is imported
2. The submodule successfully imports (line 135)
3. A `min_version` parameter is provided (triggering the version check path)
4. The parent module name is extracted (line 142-144)
5. The code attempts to access the parent module from `sys.modules` without a safety check (line 145)

This breaks user code that properly catches `ImportError` as documented, since `KeyError` is not a subclass of `ImportError` and would not be caught by exception handlers expecting the documented behavior.

## Relevant Context

The function correctly handles other error cases by raising `ImportError`:
- Line 138: Raises `ImportError` when the module cannot be imported initially
- Line 164: Raises `ImportError` when the version is too old

The existing test suite in pandas (`test_optional_dependency.py`) includes a `test_submodule` function but it always ensures both the parent and submodule exist in `sys.modules`, missing this edge case.

This scenario can occur in real-world usage when:
- Libraries use lazy loading and only the submodule gets imported
- Testing scenarios where modules are mocked partially
- Dynamic module loading where submodules are loaded independently

Link to the source code: https://github.com/pandas-dev/pandas/blob/main/pandas/compat/_optional.py

## Proposed Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -142,7 +142,11 @@ def import_optional_dependency(
     parent = name.split(".")[0]
     if parent != name:
         install_name = parent
-        module_to_get = sys.modules[install_name]
+        if install_name in sys.modules:
+            module_to_get = sys.modules[install_name]
+        else:
+            msg = f"Parent module '{install_name}' not found in sys.modules for submodule '{name}'"
+            raise ImportError(msg)
     else:
         module_to_get = module
     minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
```