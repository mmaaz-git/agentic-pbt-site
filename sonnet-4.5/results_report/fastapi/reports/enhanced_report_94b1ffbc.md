# Bug Report: fastapi.utils.deep_dict_update Non-Idempotent List Concatenation

**Target**: `fastapi.utils.deep_dict_update`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `deep_dict_update` function in FastAPI violates idempotence when updating dictionaries containing list values. Each call concatenates lists instead of merging them idempotently, causing duplicate elements to accumulate with repeated calls.

## Property-Based Test

```python
import copy
from hypothesis import given, strategies as st

from fastapi.utils import deep_dict_update

@given(
    st.dictionaries(
        st.text(min_size=1, max_size=10),
        st.lists(st.integers(), min_size=1, max_size=5),
        min_size=1,
        max_size=3
    )
)
def test_deep_dict_update_idempotence_with_lists(update_dict):
    main_dict = {}

    deep_dict_update(main_dict, update_dict)
    first_result = copy.deepcopy(main_dict)

    deep_dict_update(main_dict, update_dict)
    second_result = copy.deepcopy(main_dict)

    assert first_result == second_result, (
        f"Idempotence violated: calling deep_dict_update twice with the same "
        f"update_dict produces different results. First: {first_result}, "
        f"Second: {second_result}"
    )

if __name__ == "__main__":
    test_deep_dict_update_idempotence_with_lists()
```

<details>

<summary>
**Failing input**: `{"0": [0]}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 30, in <module>
    test_deep_dict_update_idempotence_with_lists()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 7, in test_deep_dict_update_idempotence_with_lists
    st.dictionaries(
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 23, in test_deep_dict_update_idempotence_with_lists
    assert first_result == second_result, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Idempotence violated: calling deep_dict_update twice with the same update_dict produces different results. First: {'0': [0]}, Second: {'0': [0, 0]}
Falsifying example: test_deep_dict_update_idempotence_with_lists(
    update_dict={'0': [0]},  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from fastapi.utils import deep_dict_update

# Minimal failing case from hypothesis
print("Minimal test case from Hypothesis:")
main = {}
update = {'0': [0]}

print(f"  main = {main}")
print(f"  update = {update}")
print()

deep_dict_update(main, update)
print("After 1st deep_dict_update(main, update):")
print(f"  main = {main}")
print()

deep_dict_update(main, update)
print("After 2nd deep_dict_update(main, update):")
print(f"  main = {main}")
print()

print("=" * 50)
print()

# Test basic idempotence violation
main = {}
update = {"items": [1, 2, 3]}

print("Initial state:")
print(f"  main = {main}")
print(f"  update = {update}")
print()

deep_dict_update(main, update)
print("After 1st deep_dict_update(main, update):")
print(f"  main = {main}")
print()

deep_dict_update(main, update)
print("After 2nd deep_dict_update(main, update):")
print(f"  main = {main}")
print()

deep_dict_update(main, update)
print("After 3rd deep_dict_update(main, update):")
print(f"  main = {main}")
print()

print("=" * 50)
print()

# Test with existing list
main2 = {"items": [1, 2]}
update2 = {"items": [3]}

print("Test with existing list:")
print(f"  main2 = {main2}")
print(f"  update2 = {update2}")
print()

deep_dict_update(main2, update2)
print("After 1st deep_dict_update(main2, update2):")
print(f"  main2 = {main2}")
print()

deep_dict_update(main2, update2)
print("After 2nd deep_dict_update(main2, update2):")
print(f"  main2 = {main2}")
print()

print("=" * 50)
print()

# Demonstrate that dicts are idempotent
main3 = {"config": {"debug": False}}
update3 = {"config": {"debug": True, "verbose": True}}

print("Test with nested dicts (should be idempotent):")
print(f"  main3 = {main3}")
print(f"  update3 = {update3}")
print()

deep_dict_update(main3, update3)
print("After 1st deep_dict_update(main3, update3):")
print(f"  main3 = {main3}")
print()

deep_dict_update(main3, update3)
print("After 2nd deep_dict_update(main3, update3):")
print(f"  main3 = {main3}")
print("  (Note: Same result - dicts ARE idempotent)")
```

<details>

<summary>
Lists grow with each call, violating idempotence
</summary>
```
Minimal test case from Hypothesis:
  main = {}
  update = {'0': [0]}

After 1st deep_dict_update(main, update):
  main = {'0': [0]}

After 2nd deep_dict_update(main, update):
  main = {'0': [0, 0]}

==================================================

Initial state:
  main = {}
  update = {'items': [1, 2, 3]}

After 1st deep_dict_update(main, update):
  main = {'items': [1, 2, 3]}

After 2nd deep_dict_update(main, update):
  main = {'items': [1, 2, 3, 1, 2, 3]}

After 3rd deep_dict_update(main, update):
  main = {'items': [1, 2, 3, 1, 2, 3, 1, 2, 3]}

==================================================

Test with existing list:
  main2 = {'items': [1, 2]}
  update2 = {'items': [3]}

After 1st deep_dict_update(main2, update2):
  main2 = {'items': [1, 2, 3]}

After 2nd deep_dict_update(main2, update2):
  main2 = {'items': [1, 2, 3, 3]}

==================================================

Test with nested dicts (should be idempotent):
  main3 = {'config': {'debug': False}}
  update3 = {'config': {'debug': True, 'verbose': True}}

After 1st deep_dict_update(main3, update3):
  main3 = {'config': {'debug': True, 'verbose': True}}

After 2nd deep_dict_update(main3, update3):
  main3 = {'config': {'debug': True, 'verbose': True}}
  (Note: Same result - dicts ARE idempotent)
```
</details>

## Why This Is A Bug

The `deep_dict_update` function violates the fundamental property of idempotence: `f(f(x, y), y) = f(x, y)`. This means applying the same update multiple times should produce the same result after the first application.

The function exhibits inconsistent behavior across data types:
- **Nested dictionaries**: Handled idempotently through recursive merging (lines 189-194)
- **Simple values**: Handled idempotently through replacement (lines 201-202)
- **Lists**: Handled non-idempotently through concatenation (lines 195-200)

This violates Python conventions where standard update operations (`dict.update()`, `set.update()`) are idempotent. The function name "deep_dict_update" creates a reasonable expectation of idempotent behavior.

In FastAPI's context, this function is used for OpenAPI schema generation (see `/fastapi/openapi/utils.py`). Non-idempotent behavior could cause:
- Duplicate OpenAPI tags
- Duplicate security schemes
- Duplicate server definitions
- Duplicate response schemas

These duplications would occur in retry scenarios, multiple configuration passes, or any situation where the same update is applied more than once.

## Relevant Context

The `deep_dict_update` function is located at `/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/fastapi/utils.py:187-203`

It's an internal utility function (not part of FastAPI's public API) used in OpenAPI schema generation:
- Merging additional field schemas
- Merging OpenAPI response definitions
- Merging route-specific OpenAPI extras

The function has no docstring or documentation specifying intended behavior, making the list concatenation appear to be an implementation oversight rather than intentional design.

## Proposed Fix

Replace list concatenation with idempotent merging that avoids duplicates while preserving order:

```diff
--- a/fastapi/utils.py
+++ b/fastapi/utils.py
@@ -195,7 +195,11 @@ def deep_dict_update(main_dict: Dict[Any, Any], update_dict: Dict[Any, Any]) ->
             key in main_dict
             and isinstance(main_dict[key], list)
             and isinstance(update_dict[key], list)
         ):
-            main_dict[key] = main_dict[key] + update_dict[key]
+            # Merge lists idempotently by avoiding duplicates
+            existing = main_dict[key]
+            new_items = [item for item in update_dict[key] if item not in existing]
+            main_dict[key] = existing + new_items
         else:
             main_dict[key] = value
```