# Bug Report: numpy.typing.__getattr__ NameError when accessing NBitBase

**Target**: `numpy.typing.__getattr__`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `numpy.typing.__getattr__` function crashes with a `NameError` when attempting to return `NBitBase` because the variable name is not accessible within the function's local scope, violating Python's scoping rules.

## Property-Based Test

```python
import warnings
import numpy.typing as npt
from hypothesis import given, strategies as st

@given(st.just("NBitBase"))
def test_getattr_returns_value(attr_name):
    import importlib
    # Reload the module to ensure clean state
    importlib.reload(npt)
    # Delete NBitBase from module dict to force __getattr__ call
    del npt.__dict__['NBitBase']
    # This should use __getattr__ to retrieve NBitBase
    result = getattr(npt, attr_name)
    assert result is not None

if __name__ == "__main__":
    test_getattr_returns_value()
```

<details>

<summary>
**Failing input**: `'NBitBase'`
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/23/hypo.py:13: DeprecationWarning: `NBitBase` is deprecated and will be removed from numpy.typing in the future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound, instead. (deprecated in NumPy 2.3)
  result = getattr(npt, attr_name)
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 17, in <module>
    test_getattr_returns_value()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 6, in test_getattr_returns_value
    def test_getattr_returns_value(attr_name):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/23/hypo.py", line 13, in test_getattr_returns_value
    result = getattr(npt, attr_name)
  File "/home/npc/miniconda/lib/python3.13/site-packages/numpy/typing/__init__.py", line 184, in __getattr__
    return NBitBase
           ^^^^^^^^
NameError: name 'NBitBase' is not defined
Falsifying example: test_getattr_returns_value(
    attr_name='NBitBase',
)
```
</details>

## Reproducing the Bug

```python
import numpy.typing as npt

# Delete NBitBase from the module's __dict__ to force __getattr__ to be called
del npt.__dict__['NBitBase']

# Try to access NBitBase, which should trigger __getattr__
try:
    obj = npt.NBitBase
    print(f"Successfully retrieved: {obj}")
except NameError as e:
    print(f"NameError: {e}")
```

<details>

<summary>
NameError crash when accessing NBitBase through __getattr__
</summary>
```
/home/npc/pbt/agentic-pbt/worker_/23/repo.py:8: DeprecationWarning: `NBitBase` is deprecated and will be removed from numpy.typing in the future. Use `@typing.overload` or a `TypeVar` with a scalar-type as upper bound, instead. (deprecated in NumPy 2.3)
  obj = npt.NBitBase
NameError: name 'NBitBase' is not defined
```
</details>

## Why This Is A Bug

This violates Python's expected behavior for module-level `__getattr__` as specified in PEP 562. The function attempts to return `NBitBase` directly on line 184 of `/home/npc/pbt/agentic-pbt/envs/numpy_env/lib/python3.13/site-packages/numpy/typing/__init__.py`, but this name is not available in the function's local scope.

While `NBitBase` is imported at the module level (line 160: `from numpy._typing import NBitBase`), Python's scoping rules require that functions access module-level names either through `globals()` or by importing them locally. The bare name `NBitBase` cannot be resolved within the `__getattr__` function's scope, causing a `NameError`.

This bug is currently masked in normal usage because `NBitBase` exists in the module's `__dict__`, preventing `__getattr__` from being called. However, this breaks the intended deprecation path - when NBitBase is eventually removed from regular imports (as planned per the deprecation warning), the deprecation warning system itself will fail with a crash instead of providing a helpful warning to users.

## Relevant Context

- **PEP 562 Examples**: The official PEP 562 documentation shows using `globals()` to access module-level attributes within `__getattr__`, establishing this as the expected pattern.
- **Deprecation Timeline**: NBitBase was deprecated in NumPy 2.3 (2025-05-01) and is scheduled for removal. The broken `__getattr__` implementation blocks this deprecation path.
- **Module Location**: The bug is in `/numpy/typing/__init__.py` at line 184
- **Python Version**: Tested on Python 3.13 with NumPy 2.3.0

The bug manifests when:
- NBitBase is removed from the module's `__dict__` (e.g., via `del` or module reloading)
- In future NumPy versions when NBitBase is removed from regular imports as part of the deprecation process
- In certain module reloading scenarios

## Proposed Fix

```diff
--- a/numpy/typing/__init__.py
+++ b/numpy/typing/__init__.py
@@ -181,7 +181,7 @@ def __getattr__(name: str):
             "bound, instead. (deprecated in NumPy 2.3)",
             DeprecationWarning,
             stacklevel=2,
         )
-        return NBitBase
+        return globals()['NBitBase']

     if name in __DIR_SET:
         return globals()[name]
```