# Bug Report: fastapi.dependencies get_flat_dependant Circular Dependency

**Target**: `fastapi.dependencies.utils.get_flat_dependant`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`get_flat_dependant()` does not detect or handle circular dependencies, leading to RecursionError when dependency graphs contain cycles.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_flat_dependant


def test_circular_dependency_detection():
    dep1 = Dependant(call=lambda: "dep1", name="dep1")
    dep2 = Dependant(call=lambda: "dep2", name="dep2", dependencies=[dep1])
    dep1.dependencies.append(dep2)

    flat = get_flat_dependant(dep1)
    assert isinstance(flat, Dependant)
```

**Failing input**: Two dependants that reference each other (dep1 -> dep2 -> dep1)

## Reproducing the Bug

```python
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_flat_dependant

dep1 = Dependant(call=lambda: "dep1", name="dep1")
dep2 = Dependant(call=lambda: "dep2", name="dep2", dependencies=[dep1])
dep1.dependencies.append(dep2)

flat = get_flat_dependant(dep1)
```

Running this code results in:
```
RecursionError: maximum recursion depth exceeded
```

## Why This Is A Bug

The function recursively processes dependencies without checking if a dependant has already been visited in the current recursion path. While `skip_repeats=True` uses a `visited` list to avoid processing the same cache_key twice, it doesn't prevent infinite recursion when there's a cycle where dependants have different cache_keys but still form a circular reference.

## Fix

The fix should track the current recursion path separately from the global visited set. Here's a high-level approach:

1. Add a `recursion_stack` parameter to track the current path
2. Before recursing into a sub_dependant, check if it's already in the recursion_stack
3. If found, skip it to break the cycle

Example fix:

```diff
def get_flat_dependant(
    dependant: Dependant,
    *,
    skip_repeats: bool = False,
    visited: Optional[List[CacheKey]] = None,
+   recursion_stack: Optional[set] = None,
) -> Dependant:
    if visited is None:
        visited = []
+   if recursion_stack is None:
+       recursion_stack = set()
+
+   if id(dependant) in recursion_stack:
+       return Dependant(use_cache=dependant.use_cache, path=dependant.path)
+
+   recursion_stack.add(id(dependant))
    visited.append(dependant.cache_key)

    flat_dependant = Dependant(
        path_params=dependant.path_params.copy(),
        query_params=dependant.query_params.copy(),
        header_params=dependant.header_params.copy(),
        cookie_params=dependant.cookie_params.copy(),
        body_params=dependant.body_params.copy(),
        security_requirements=dependant.security_requirements.copy(),
        use_cache=dependant.use_cache,
        path=dependant.path,
    )
    for sub_dependant in dependant.dependencies:
        if skip_repeats and sub_dependant.cache_key in visited:
            continue
        flat_sub = get_flat_dependant(
            sub_dependant,
            skip_repeats=skip_repeats,
            visited=visited,
+           recursion_stack=recursion_stack,
        )
        flat_dependant.path_params.extend(flat_sub.path_params)
        flat_dependant.query_params.extend(flat_sub.query_params)
        flat_dependant.header_params.extend(flat_sub.header_params)
        flat_dependant.cookie_params.extend(flat_sub.cookie_params)
        flat_dependant.body_params.extend(flat_sub.body_params)
        flat_dependant.security_requirements.extend(flat_sub.security_requirements)
+
+   recursion_stack.remove(id(dependant))
    return flat_dependant
```