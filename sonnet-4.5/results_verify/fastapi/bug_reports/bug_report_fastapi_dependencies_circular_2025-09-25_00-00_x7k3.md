# Bug Report: FastAPI Dependencies get_flat_dependant Infinite Recursion

**Target**: `fastapi.dependencies.utils.get_flat_dependant`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_flat_dependant` function causes infinite recursion (RecursionError) when processing circular dependencies with `skip_repeats=False`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_flat_dependant
import sys


@st.composite
def circular_dependant_pair(draw):
    def func_a():
        return "a"
    def func_b():
        return "b"

    dep_a = Dependant(call=func_a, name="dep_a")
    dep_b = Dependant(call=func_b, name="dep_b")

    create_cycle = draw(st.booleans())

    if create_cycle:
        dep_a.dependencies.append(dep_b)
        dep_b.dependencies.append(dep_a)

    return dep_a, create_cycle


@given(circular_dependant_pair())
@settings(max_examples=50, deadline=1000)
def test_get_flat_dependant_no_infinite_recursion(data):
    dep, has_cycle = data

    old_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(100)

    try:
        flat = get_flat_dependant(dep, skip_repeats=False)
        assert isinstance(flat, Dependant)
    except RecursionError:
        if has_cycle:
            raise AssertionError("Infinite recursion with circular dependencies")
        raise
    finally:
        sys.setrecursionlimit(old_limit)
```

**Failing input**: Two dependants A and B where A depends on B and B depends on A

## Reproducing the Bug

```python
from fastapi.dependencies.models import Dependant
from fastapi.dependencies.utils import get_flat_dependant
import sys

sys.setrecursionlimit(50)

def dep_a():
    return "a"

def dep_b():
    return "b"

dependant_a = Dependant(call=dep_a, name="dep_a")
dependant_b = Dependant(call=dep_b, name="dep_b")

dependant_a.dependencies.append(dependant_b)
dependant_b.dependencies.append(dependant_a)

flat = get_flat_dependant(dependant_a, skip_repeats=False)
```

## Why This Is A Bug

The `get_flat_dependant` function maintains a `visited` list to track processed dependencies, but only checks this list when `skip_repeats=True`. When `skip_repeats=False`, the function still modifies the `visited` list but never checks it, leading to infinite recursion with circular dependencies.

The issue is in `fastapi/dependencies/utils.py`:
- Line 185: `visited.append(dependant.cache_key)` - Always modifies visited
- Line 198: `if skip_repeats and sub_dependant.cache_key in visited:` - Only checks visited when skip_repeats=True

When `skip_repeats=False`, circular dependencies cause infinite recursion because:
1. The visited list is populated but never consulted
2. Each recursive call processes the same dependencies again
3. The recursion never terminates until RecursionError

While circular dependencies are a design smell, the function should handle them gracefully rather than crashing. The current behavior is inconsistent: the `visited` parameter exists and is maintained, suggesting the intention was to prevent infinite loops, but the check is conditional on `skip_repeats`.

## Fix

```diff
def get_flat_dependant(
    dependant: Dependant,
    *,
    skip_repeats: bool = False,
    visited: Optional[List[CacheKey]] = None,
) -> Dependant:
    if visited is None:
        visited = []
+   # Always check for cycles to prevent infinite recursion
+   if dependant.cache_key in visited and not skip_repeats:
+       # For skip_repeats=False, we allow repeats but must prevent infinite loops
+       # Return an empty dependant to break the cycle while continuing flattening
+       return Dependant(
+           use_cache=dependant.use_cache,
+           path=dependant.path,
+       )
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
            sub_dependant, skip_repeats=skip_repeats, visited=visited
        )
        flat_dependant.path_params.extend(flat_sub.path_params)
        flat_dependant.query_params.extend(flat_sub.query_params)
        flat_dependant.header_params.extend(flat_sub.header_params)
        flat_dependant.cookie_params.extend(flat_sub.cookie_params)
        flat_dependant.body_params.extend(flat_sub.body_params)
        flat_dependant.security_requirements.extend(flat_sub.security_requirements)
    return flat_dependant
```

Alternatively, a simpler fix would be to always check `visited` regardless of `skip_repeats`:

```diff
    for sub_dependant in dependant.dependencies:
-       if skip_repeats and sub_dependant.cache_key in visited:
+       if sub_dependant.cache_key in visited and (skip_repeats or True):
            continue
```

Or more clearly:

```diff
    for sub_dependant in dependant.dependencies:
+       # Always skip if already visited to prevent infinite recursion
+       if sub_dependant.cache_key in visited:
-       if skip_repeats and sub_dependant.cache_key in visited:
            continue
```

This ensures that circular dependencies never cause infinite recursion, while still allowing repeated processing when intended (by not having the same cache_key).