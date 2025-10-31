# Bug Report: pandas.compat._optional.get_version Returns Non-String Values Violating Type Contract

**Target**: `pandas.compat._optional.get_version`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `get_version()` function returns non-string values when a module's `__version__` attribute is not a string, violating its return type annotation `-> str` and causing AttributeError crashes for psycopg2 modules.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.compat._optional import get_version
import types


@given(st.one_of(st.integers(), st.floats(), st.lists(st.integers()), st.none()))
@settings(max_examples=100)
def test_get_version_with_non_string_version(version_value):
    mock_module = types.ModuleType("test_module")
    mock_module.__version__ = version_value

    result = get_version(mock_module)
    assert isinstance(result, str)


if __name__ == "__main__":
    test_get_version_with_non_string_version()
```

<details>

<summary>
**Failing input**: `version_value=0`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 17, in <module>
  |     test_get_version_with_non_string_version()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 7, in test_get_version_with_non_string_version
  |     @settings(max_examples=100)
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 13, in test_get_version_with_non_string_version
    |     assert isinstance(result, str)
    |            ~~~~~~~~~~^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_get_version_with_non_string_version(
    |     version_value=0,  # or any other generated value
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 12, in test_get_version_with_non_string_version
    |     result = get_version(mock_module)
    |   File "/home/npc/miniconda/lib/python3.13/site-packages/pandas/compat/_optional.py", line 78, in get_version
    |     raise ImportError(f"Can't determine version for {module.__name__}")
    | ImportError: Can't determine version for test_module
    | Falsifying example: test_get_version_with_non_string_version(
    |     version_value=None,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from pandas.compat._optional import get_version
import types

# Test 1: Non-string __version__ returns non-string value
mock_module = types.ModuleType("test_module")
mock_module.__version__ = 0

result = get_version(mock_module)
print(f"Test 1 - Non-string version:")
print(f"  Result: {result}")
print(f"  Type: {type(result)}")
print(f"  Expected type: str")
print(f"  Type matches expectation: {isinstance(result, str)}")
print()

# Test 2: psycopg2 module with non-string __version__ causes crash
psycopg2_module = types.ModuleType("psycopg2")
psycopg2_module.__version__ = 42

print("Test 2 - psycopg2 with non-string version:")
try:
    result = get_version(psycopg2_module)
    print(f"  Result: {result}")
    print(f"  Type: {type(result)}")
except AttributeError as e:
    print(f"  Error occurred: AttributeError: {e}")
    print(f"  This happens at line 81 where version.split() is called")
```

<details>

<summary>
AttributeError crash for psycopg2 modules with non-string __version__
</summary>
```
Test 1 - Non-string version:
  Result: 0
  Type: <class 'int'>
  Expected type: str
  Type matches expectation: False

Test 2 - psycopg2 with non-string version:
  Error occurred: AttributeError: 'int' object has no attribute 'split'
  This happens at line 81 where version.split() is called
```
</details>

## Why This Is A Bug

This bug violates the function's type contract and implementation assumptions in multiple ways:

1. **Type Contract Violation**: The function signature explicitly declares `-> str` as the return type (line 74), creating a contract that callers can rely on receiving a string. However, when `__version__` is a non-string value like an integer or float, the function returns that non-string value directly, violating this contract.

2. **Runtime Crash for psycopg2**: The function contains special handling for psycopg2 modules at line 81 (`version = version.split()[0]`) which assumes `version` is a string. When psycopg2 has a non-string `__version__`, this causes an immediate AttributeError crash.

3. **Downstream Impact**: The function is called by `import_optional_dependency()` at line 150, which then passes the result to `Version(version)` for version comparison. While Version() may handle some non-string inputs, the type contract violation can cause unexpected behavior in type-checked codebases.

## Relevant Context

The `get_version()` function is located in `/home/npc/miniconda/lib/python3.13/site-packages/pandas/compat/_optional.py` and is part of pandas' internal dependency management system. While this is a private module (indicated by the underscore prefix), it's used internally by pandas to check versions of optional dependencies.

According to Python conventions (PEP 396), the `__version__` attribute should be a string, but Python doesn't enforce this at runtime. Real-world packages occasionally deviate from this convention, and robust code should handle such cases gracefully rather than crashing or violating type contracts.

The bug manifests in two scenarios:
- General case: Returns non-string values when expecting strings
- Specific case: Crashes with AttributeError for psycopg2 modules with non-string versions

## Proposed Fix

```diff
--- a/pandas/compat/_optional.py
+++ b/pandas/compat/_optional.py
@@ -76,6 +76,8 @@ def get_version(module: types.ModuleType) -> str:

     if version is None:
         raise ImportError(f"Can't determine version for {module.__name__}")
+    if not isinstance(version, str):
+        version = str(version)
     if module.__name__ == "psycopg2":
         # psycopg2 appends " (dt dec pq3 ext lo64)" to it's version
         version = version.split()[0]
```