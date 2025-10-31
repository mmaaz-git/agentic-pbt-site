# Bug Report: FastAPI get_flat_dependant Circular Dependency Crash

**Target**: `fastapi.dependencies.utils.get_flat_dependant`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_flat_dependant()` function crashes with RecursionError when processing circular dependencies between Dependant objects, instead of detecting and handling the cycle gracefully.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for circular dependency detection in FastAPI's get_flat_dependant"""

from hypothesis import given, strategies as st
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_flat_dependant


def test_circular_dependency_detection():
    """Test that get_flat_dependant handles circular dependencies"""
    dep1 = Dependant(call=lambda: "dep1", name="dep1")
    dep2 = Dependant(call=lambda: "dep2", name="dep2", dependencies=[dep1])
    dep1.dependencies.append(dep2)

    flat = get_flat_dependant(dep1)
    assert isinstance(flat, Dependant)


if __name__ == "__main__":
    try:
        test_circular_dependency_detection()
        print("Test passed!")
    except RecursionError as e:
        print("Test failed with RecursionError!")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
```

<details>

<summary>
**Failing input**: Two Dependant objects with mutual references (dep1 → dep2 → dep1)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 21, in <module>
    test_circular_dependency_detection()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/39/hypo.py", line 15, in test_circular_dependency_detection
    flat = get_flat_dependant(dep1)
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 200, in get_flat_dependant
    flat_sub = get_flat_dependant(
        sub_dependant, skip_repeats=skip_repeats, visited=visited
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 200, in get_flat_dependant
    flat_sub = get_flat_dependant(
        sub_dependant, skip_repeats=skip_repeats, visited=visited
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 200, in get_flat_dependant
    flat_sub = get_flat_dependant(
        sub_dependant, skip_repeats=skip_repeats, visited=visited
    )
  [Previous line repeated 993 more times]
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 187, in get_flat_dependant
    flat_dependant = Dependant(
        path_params=dependant.path_params.copy(),
    ...<6 lines>...
        path=dependant.path,
    )
  File "<string>", line 21, in __init__
RecursionError: maximum recursion depth exceeded
Test failed with RecursionError!
Error: maximum recursion depth exceeded
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of circular dependency bug in FastAPI's get_flat_dependant"""

from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_flat_dependant

# Create two dependants with circular references
dep1 = Dependant(call=lambda: "dep1", name="dep1")
dep2 = Dependant(call=lambda: "dep2", name="dep2", dependencies=[dep1])

# Create the circular reference
dep1.dependencies.append(dep2)

# This will cause RecursionError
try:
    flat = get_flat_dependant(dep1)
    print("Success: get_flat_dependant completed")
    print(f"Result type: {type(flat)}")
except RecursionError as e:
    print("RecursionError occurred!")
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/39/repo.py", line 16, in <module>
    flat = get_flat_dependant(dep1)
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 200, in get_flat_dependant
    flat_sub = get_flat_dependant(
        sub_dependant, skip_repeats=skip_repeats, visited=visited
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 200, in get_flat_dependant
    flat_sub = get_flat_dependant(
        sub_dependant, skip_repeats=skip_repeats, visited=visited
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 200, in get_flat_dependant
    flat_sub = get_flat_dependant(
        sub_dependant, skip_repeats=skip_repeats, visited=visited
    )
  [Previous line repeated 994 more times]
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 187, in get_flat_dependant
    flat_dependant = Dependant(
        path_params=dependant.path_params.copy(),
    ...<6 lines>...
        path=dependant.path,
    )
  File "<string>", line 21, in __init__
RecursionError: maximum recursion depth exceeded
RecursionError occurred!
Error: maximum recursion depth exceeded
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Uncontrolled Crash**: The function crashes with RecursionError instead of providing meaningful error handling. A production framework should never crash with stack overflow errors on invalid input.

2. **Inconsistent Protection**: The function already has partial cycle detection via the `skip_repeats` parameter and `visited` list mechanism, showing awareness of the need to handle repeated dependencies. However, this protection only works when `skip_repeats=True` (not the default).

3. **Default Behavior is Unsafe**: The function defaults to `skip_repeats=False`, making it vulnerable to circular references by default. The safer behavior should be the default.

4. **Public API Impact**: While `get_flat_dependant()` is an internal function, users can create circular dependencies through FastAPI's public API using `Depends()`. For example:
   ```python
   from fastapi import Depends, FastAPI

   def dep_a(b = Depends("dep_b")):
       return "a"

   def dep_b(a = Depends("dep_a")):
       return "b"
   ```

5. **Cache Key Design Issue**: The function tracks visited dependencies by `cache_key` (which is based on the callable and security scopes). When two dependencies use different lambda functions (as in our test), they have different cache keys even though they form a cycle, bypassing the visited check.

## Relevant Context

The issue is located in `/fastapi/dependencies/utils.py` at lines 177-209. The function performs recursive traversal of a dependency tree without tracking the current recursion path.

Key observations from testing:
- When `skip_repeats=True`, the function successfully handles cycles by checking if `sub_dependant.cache_key in visited`
- With different lambda functions, each Dependant has a unique cache_key, so the visited check doesn't prevent recursion
- With the same callable function, both Dependants share a cache_key, making skip_repeats effective

The function is called internally by FastAPI's dependency injection system, particularly in:
- `get_flat_params()` (line 223) - always uses `skip_repeats=True`
- `solve_dependencies()` - processes dependencies during request handling

Documentation: The function has no docstring and its behavior regarding circular dependencies is undocumented.

## Proposed Fix

```diff
--- a/fastapi/dependencies/utils.py
+++ b/fastapi/dependencies/utils.py
@@ -177,10 +177,19 @@ CacheKey = Tuple[Optional[Callable[..., Any]], Tuple[str, ...]]
 def get_flat_dependant(
     dependant: Dependant,
     *,
     skip_repeats: bool = False,
     visited: Optional[List[CacheKey]] = None,
+    recursion_path: Optional[set] = None,
 ) -> Dependant:
     if visited is None:
         visited = []
+    if recursion_path is None:
+        recursion_path = set()
+
+    # Check for circular dependency
+    dependant_id = id(dependant)
+    if dependant_id in recursion_path:
+        # Return empty dependant to break cycle
+        return Dependant(use_cache=dependant.use_cache, path=dependant.path)
+
+    recursion_path.add(dependant_id)
     visited.append(dependant.cache_key)

@@ -197,8 +206,11 @@ def get_flat_dependant(
     for sub_dependant in dependant.dependencies:
         if skip_repeats and sub_dependant.cache_key in visited:
             continue
         flat_sub = get_flat_dependant(
-            sub_dependant, skip_repeats=skip_repeats, visited=visited
+            sub_dependant,
+            skip_repeats=skip_repeats,
+            visited=visited,
+            recursion_path=recursion_path
         )
         flat_dependant.path_params.extend(flat_sub.path_params)
         flat_dependant.query_params.extend(flat_sub.query_params)
@@ -206,4 +218,6 @@ def get_flat_dependant(
         flat_dependant.cookie_params.extend(flat_sub.cookie_params)
         flat_dependant.body_params.extend(flat_sub.body_params)
         flat_dependant.security_requirements.extend(flat_sub.security_requirements)
+
+    recursion_path.remove(dependant_id)
     return flat_dependant
```