# Bug Report: fastapi.dependencies.utils.get_flat_dependant Infinite Recursion with Circular Dependencies

**Target**: `fastapi.dependencies.utils.get_flat_dependant`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `get_flat_dependant` function crashes with RecursionError when processing circular dependencies with `skip_repeats=False`, despite maintaining a visited list that could prevent infinite recursion.

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

<details>

<summary>
**Failing input**: Two Dependant objects with circular reference (dep_a depends on dep_b, dep_b depends on dep_a)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 35, in test_get_flat_dependant_no_infinite_recursion
    flat = get_flat_dependant(dep, skip_repeats=False)
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
  [Previous line repeated 85 more times]
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 187, in get_flat_dependant
    flat_dependant = Dependant(
        path_params=dependant.path_params.copy(),
    ...<6 lines>...
        path=dependant.path,
    )
  File "<string>", line 21, in __init__
RecursionError: maximum recursion depth exceeded

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 45, in <module>
    test_get_flat_dependant_no_infinite_recursion()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 27, in test_get_flat_dependant_no_infinite_recursion
    @settings(max_examples=50, deadline=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/36/hypo.py", line 39, in test_get_flat_dependant_no_infinite_recursion
    raise AssertionError("Infinite recursion with circular dependencies")
AssertionError: Infinite recursion with circular dependencies
Falsifying example: test_get_flat_dependant_no_infinite_recursion(
    data=(Dependant(path_params=[],
      query_params=[],
      header_params=[],
      cookie_params=[],
      body_params=[],
      dependencies=[Dependant(path_params=[],
        query_params=[],
        header_params=[],
        cookie_params=[],
        body_params=[],
        dependencies=[Dependant(...)],
        security_requirements=[],
        name='dep_b',
        call=func_b,
        request_param_name=None,
        websocket_param_name=None,
        http_connection_param_name=None,
        response_param_name=None,
        background_tasks_param_name=None,
        security_scopes_param_name=None,
        security_scopes=None,
        use_cache=True,
        path=None)],
      security_requirements=[],
      name='dep_a',
      call=func_a,
      request_param_name=None,
      websocket_param_name=None,
      http_connection_param_name=None,
      response_param_name=None,
      background_tasks_param_name=None,
      security_scopes_param_name=None,
      security_scopes=None,
      use_cache=True,
      path=None),
     True),
)
```
</details>

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

<details>

<summary>
RecursionError: maximum recursion depth exceeded
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/36/repo.py", line 19, in <module>
    flat = get_flat_dependant(dependant_a, skip_repeats=False)
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
  [Previous line repeated 44 more times]
  File "/home/npc/miniconda/lib/python3.13/site-packages/fastapi/dependencies/utils.py", line 187, in get_flat_dependant
    flat_dependant = Dependant(
        path_params=dependant.path_params.copy(),
    ...<6 lines>...
        path=dependant.path,
    )
  File "<string>", line 21, in __init__
RecursionError: maximum recursion depth exceeded
```
</details>

## Why This Is A Bug

The `get_flat_dependant` function in `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/dependencies/utils.py` contains a logic error that causes infinite recursion when processing circular dependencies with `skip_repeats=False`.

The function maintains a `visited` list to track processed dependencies (line 185: `visited.append(dependant.cache_key)`), but only checks this list when `skip_repeats=True` (line 198: `if skip_repeats and sub_dependant.cache_key in visited:`). This inconsistency causes the function to:

1. Always append to the `visited` list regardless of the `skip_repeats` flag
2. Only check the `visited` list to prevent re-processing when `skip_repeats=True`
3. When `skip_repeats=False`, the function recursively processes the same circular dependencies indefinitely

The code contradicts its apparent intention: the `visited` parameter exists to prevent infinite loops in recursive traversal, but the conditional check makes it ineffective when `skip_repeats=False`. This violates the principle that recursive functions processing graph-like structures should always have cycle detection to prevent stack overflow, regardless of other processing flags.

## Relevant Context

The bug occurs in FastAPI's dependency injection system, which is a core feature used to manage and resolve dependencies in API endpoints. The `get_flat_dependant` function is responsible for flattening nested dependencies into a single structure that can be processed efficiently.

The function is called from multiple places in FastAPI:
- Line 223 in the same file: `get_flat_params` calls it with `skip_repeats=True`
- Various other dependency resolution contexts

While circular dependencies are generally considered an anti-pattern in dependency injection systems, the framework should handle them gracefully rather than crashing. Many dependency injection frameworks either detect cycles and raise a descriptive error or allow controlled circular resolution.

Code location: `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/dependencies/utils.py:177-209`

## Proposed Fix

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
+   if dependant.cache_key in visited:
+       # Already processing this dependency - avoid infinite recursion
+       if skip_repeats:
+           # When skip_repeats=True, return empty dependant
+           return Dependant(
+               use_cache=dependant.use_cache,
+               path=dependant.path,
+           )
+       # When skip_repeats=False, we still need to prevent infinite loops
+       # Return a shallow copy without recursing into dependencies again
+       return Dependant(
+           path_params=dependant.path_params.copy(),
+           query_params=dependant.query_params.copy(),
+           header_params=dependant.header_params.copy(),
+           cookie_params=dependant.cookie_params.copy(),
+           body_params=dependant.body_params.copy(),
+           security_requirements=dependant.security_requirements.copy(),
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
-       if skip_repeats and sub_dependant.cache_key in visited:
-           continue
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