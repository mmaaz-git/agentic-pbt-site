# Bug Report: xarray.backends.plugins.sort_backends Mutates Input Dictionary

**Target**: `xarray.backends.plugins.sort_backends`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `sort_backends` function mutates its input dictionary by removing entries that match names in `NETCDF_BACKENDS_ORDER`, violating the expected immutability of function arguments.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from xarray.backends.plugins import sort_backends
from xarray.backends.common import BackendEntrypoint

class MockBackend(BackendEntrypoint):
    def open_dataset(self, *args, **kwargs):
        pass
    def guess_can_open(self, *args, **kwargs):
        return False

@given(
    has_scipy=st.booleans(),
    num_custom=st.integers(min_value=0, max_value=5)
)
def test_sort_backends_immutable(has_scipy, num_custom):
    backend_dict = {}
    if has_scipy:
        backend_dict["scipy"] = MockBackend
    for i in range(num_custom):
        backend_dict[f"custom{i}"] = MockBackend

    original_keys = set(backend_dict.keys())
    result = sort_backends(backend_dict)

    assert set(backend_dict.keys()) == original_keys
```

**Failing input**: `has_scipy=True, num_custom=0`

## Reproducing the Bug

```python
from xarray.backends.plugins import sort_backends
from xarray.backends.common import BackendEntrypoint

class MockBackend(BackendEntrypoint):
    def open_dataset(self, *args, **kwargs):
        pass
    def guess_can_open(self, *args, **kwargs):
        return False

backend_dict = {"scipy": MockBackend, "custom": MockBackend}
print(f"Before: {list(backend_dict.keys())}")

result = sort_backends(backend_dict)

print(f"After: {list(backend_dict.keys())}")
print(f"Result: {list(result.keys())}")
```

**Output:**
```
Before: ['scipy', 'custom']
After: ['custom']
Result: ['scipy', 'custom']
```

## Why This Is A Bug

The bug is on lines 95-97 of `plugins.py`:

```python
for be_name in NETCDF_BACKENDS_ORDER:
    if backend_entrypoints.get(be_name) is not None:
        ordered_backends_entrypoints[be_name] = backend_entrypoints.pop(be_name)
```

The function uses `.pop(be_name)` which removes the key from the input dictionary. This violates the principle of least surprise - functions should not modify their inputs unless clearly documented or the modification is the primary purpose.

This bug can cause issues when:
1. The caller needs to use the input dictionary after calling `sort_backends`
2. The same dictionary is used in multiple places
3. The dictionary is shared between threads

## Fix

```diff
--- a/xarray/backends/plugins.py
+++ b/xarray/backends/plugins.py
@@ -91,10 +91,10 @@ def set_missing_parameters(
 def sort_backends(
     backend_entrypoints: dict[str, type[BackendEntrypoint]],
 ) -> dict[str, type[BackendEntrypoint]]:
     ordered_backends_entrypoints = {}
     for be_name in NETCDF_BACKENDS_ORDER:
         if backend_entrypoints.get(be_name) is not None:
-            ordered_backends_entrypoints[be_name] = backend_entrypoints.pop(be_name)
+            ordered_backends_entrypoints[be_name] = backend_entrypoints[be_name]
     ordered_backends_entrypoints.update(
-        {name: backend_entrypoints[name] for name in sorted(backend_entrypoints)}
+        {name: backend_entrypoints[name] for name in sorted(backend_entrypoints) if name not in ordered_backends_entrypoints}
     )
     return ordered_backends_entrypoints
```

This fix:
1. Changes `.pop(be_name)` to `[be_name]` to avoid mutation
2. Adds a filter in the dict comprehension to avoid duplicate entries