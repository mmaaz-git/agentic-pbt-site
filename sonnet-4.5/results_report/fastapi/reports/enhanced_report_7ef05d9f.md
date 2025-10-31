# Bug Report: FastAPI get_path_param_names Accepts Empty Parameter Names

**Target**: `fastapi.utils.get_path_param_names`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `get_path_param_names` function incorrectly accepts and returns empty strings when paths contain empty braces `{}`, violating the semantic requirement that path parameters must have non-empty names to be referenceable in code.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for FastAPI get_path_param_names empty parameter names bug"""

from hypothesis import given, strategies as st, example
from fastapi.utils import get_path_param_names

@given(st.text())
@example("{}")  # Force it to test with empty braces
@example("/users/{}/posts")
@example("/{}/{}/{}")
def test_get_path_param_names_no_empty_strings(path):
    result = get_path_param_names(path)
    for name in result:
        assert name != '', f"Empty parameter name found for path: {path!r}"

if __name__ == "__main__":
    test_get_path_param_names_no_empty_strings()
```

<details>

<summary>
**Failing input**: `'{}'`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/30
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_get_path_param_names_no_empty_strings FAILED               [100%]

=================================== FAILURES ===================================
__________________ test_get_path_param_names_no_empty_strings __________________
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 8, in test_get_path_param_names_no_empty_strings
  |     @example("{}")  # Force it to test with empty braces
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures in explicit examples. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 14, in test_get_path_param_names_no_empty_strings
    |     assert name != '', f"Empty parameter name found for path: {path!r}"
    | AssertionError: Empty parameter name found for path: '{}'
    | assert '' != ''
    | Falsifying explicit example: test_get_path_param_names_no_empty_strings(
    |     path='{}',
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 14, in test_get_path_param_names_no_empty_strings
    |     assert name != '', f"Empty parameter name found for path: {path!r}"
    | AssertionError: Empty parameter name found for path: '/users/{}/posts'
    | assert '' != ''
    | Falsifying explicit example: test_get_path_param_names_no_empty_strings(
    |     path='/users/{}/posts',
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/30/hypo.py", line 14, in test_get_path_param_names_no_empty_strings
    |     assert name != '', f"Empty parameter name found for path: {path!r}"
    | AssertionError: Empty parameter name found for path: '/{}/{}/{}'
    | assert '' != ''
    | Falsifying explicit example: test_get_path_param_names_no_empty_strings(
    |     path='/{}/{}/{}',
    | )
    +------------------------------------
=========================== short test summary info ============================
FAILED hypo.py::test_get_path_param_names_no_empty_strings - ExceptionGroup: ...
============================== 1 failed in 0.29s ===============================
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction case for FastAPI get_path_param_names empty parameter bug"""

from fastapi.utils import get_path_param_names

# Test with empty braces in path
path = "/users/{}/posts"
result = get_path_param_names(path)

print(f"Path: {path}")
print(f"Result: {result}")
print(f"Empty string in result: {'' in result}")

# Test with multiple empty parameters
path2 = "/{}/{}/{}"
result2 = get_path_param_names(path2)
print(f"\nPath: {path2}")
print(f"Result: {result2}")

# Test with mixed empty and named parameters
path3 = "/mixed/{id}/{}/end"
result3 = get_path_param_names(path3)
print(f"\nPath: {path3}")
print(f"Result: {result3}")

# Demonstrate the issue - empty strings can't be used as function parameters
print("\nWhy this is a problem:")
print("- Python function parameters cannot have empty names")
print("- Empty strings will never match any function parameter in the route handler")
print("- This violates the semantic expectation that path parameters have valid names")

# Assert that the bug exists
assert '' in result, "Bug not reproduced - empty string should be in result"
print("\nBug confirmed: Empty string is present in path parameter names")
```

<details>

<summary>
Empty parameter names are returned for paths with empty braces
</summary>
```
Path: /users/{}/posts
Result: {''}
Empty string in result: True

Path: /{}/{}/{}
Result: {''}

Path: /mixed/{id}/{}/end
Result: {'', 'id'}

Why this is a problem:
- Python function parameters cannot have empty names
- Empty strings will never match any function parameter in the route handler
- This violates the semantic expectation that path parameters have valid names

Bug confirmed: Empty string is present in path parameter names
```
</details>

## Why This Is A Bug

This violates expected behavior in several critical ways:

1. **Semantic Contract Violation**: Path parameters must have valid names to be referenced in function signatures. The OpenAPI specification and FastAPI's design both require that path parameters correspond to named entities that can be referenced in code.

2. **Python Language Constraints**: Python function parameters must be valid identifiers, and empty strings are not valid Python identifiers. Therefore, an empty path parameter name can never match any function parameter, making it impossible to use.

3. **Silent Failure**: The function accepts invalid input without warning. In `fastapi/dependencies/utils.py:284`, the code checks `is_path_param = param_name in path_param_names`. When `path_param_names` contains an empty string, it will never match any actual function parameter name, causing the parameter to be silently ignored rather than properly validated.

4. **Framework Design Principles**: FastAPI emphasizes type safety, automatic validation, and clear error messages. Accepting empty parameter names contradicts these principles by allowing semantically invalid paths to pass through without validation.

5. **Documentation Mismatch**: All FastAPI documentation examples show named parameters like `{item_id}`, `{user_id}`, etc. There are no examples of empty parameters because they are not intended to be supported.

## Relevant Context

The bug is located in `/fastapi/utils.py` at lines 59-60:
```python
def get_path_param_names(path: str) -> Set[str]:
    return set(re.findall("{(.*?)}", path))
```

The regex pattern `{(.*?)}` matches any content between braces, including empty content. When braces contain nothing (`{}`), the captured group is an empty string.

This function is used in `fastapi/dependencies/utils.py:273` to determine which function parameters are path parameters. The empty strings in the result set will never match actual function parameter names, leading to unmatched path parameters.

Key documentation references:
- FastAPI Path Parameters: https://fastapi.tiangolo.com/tutorial/path-params/
- OpenAPI Specification v3.1.0 Path Templating: https://spec.openapis.org/oas/v3.1.0#path-templating

## Proposed Fix

```diff
 def get_path_param_names(path: str) -> Set[str]:
-    return set(re.findall("{(.*?)}", path))
+    return set(name for name in re.findall("{(.*?)}", path) if name)
```

This simple fix filters out empty strings from the result set, ensuring that only valid, non-empty parameter names are returned. This maintains backward compatibility for all valid use cases while preventing the acceptance of semantically invalid empty parameter names.