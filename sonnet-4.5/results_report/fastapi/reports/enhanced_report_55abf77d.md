# Bug Report: fastapi.dependencies.utils.analyze_param Path Parameter Default Value Inconsistency

**Target**: `fastapi.dependencies.utils.analyze_param`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `analyze_param` function enforces the constraint that "Path parameters cannot have default values" inconsistently. When using `Annotated[int, Path()]` it raises an AssertionError, but when using plain type annotations like `int`, it silently accepts the default value (though doesn't actually use it).

## Property-Based Test

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, example
from typing import Annotated
import inspect
from fastapi.dependencies.utils import analyze_param
from fastapi import Path

@given(st.integers())
@example(42)  # Include the specific failing case
@example(0)
@example(-1)
def test_path_param_default_consistency(default_value):
    """
    Property: Path parameter handling should be consistent regardless
    of whether we use plain annotations or Annotated with Path().

    Both should either:
    1. Allow defaults (consistent behavior), or
    2. Reject defaults (consistent behavior)

    Currently: Plain annotation silently allows defaults,
               Annotated raises AssertionError
    """

    # Case 1: Plain annotation with default value
    try:
        details1 = analyze_param(
            param_name="item_id",
            annotation=int,
            value=default_value,
            is_path_param=True
        )
        case1_succeeded = True
        case1_exception = None
    except Exception as e:
        case1_succeeded = False
        case1_exception = e

    # Case 2: Annotated with Path() and default value
    path_info = Path()
    path_info.annotation = int

    try:
        details2 = analyze_param(
            param_name="item_id",
            annotation=Annotated[int, path_info],
            value=default_value,
            is_path_param=True
        )
        case2_succeeded = True
        case2_exception = None
    except Exception as e:
        case2_succeeded = False
        case2_exception = e

    # Property assertion: Both cases should behave the same
    assert case1_succeeded == case2_succeeded, (
        f"Inconsistent behavior for default value {default_value}:\n"
        f"  Plain annotation: {'succeeded' if case1_succeeded else f'failed with {type(case1_exception).__name__}'}\n"
        f"  Annotated[int, Path()]: {'succeeded' if case2_succeeded else f'failed with {type(case2_exception).__name__}'}\n"
    )

    # If both succeeded, verify they produce the same type of field info
    if case1_succeeded and case2_succeeded:
        assert type(details1.field.field_info) == type(details2.field.field_info), (
            f"Different field info types for default value {default_value}:\n"
            f"  Plain: {type(details1.field.field_info)}\n"
            f"  Annotated: {type(details2.field.field_info)}"
        )

if __name__ == "__main__":
    # Run the test
    test_path_param_default_consistency()
```

<details>

<summary>
**Failing input**: `42`, `0`, `-1` (any non-empty integer value)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 76, in <module>
  |     test_path_param_default_consistency()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 12, in test_path_param_default_consistency
  |     @example(42)  # Include the specific failing case
  |                    ^^^
  |   File "/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
  |     _raise_to_user(errors, state.settings, [], " in explicit examples")
  |     ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  |   File "/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 3 distinct failures in explicit examples. (3 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 60, in test_path_param_default_consistency
    |     assert case1_succeeded == case2_succeeded, (
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Inconsistent behavior for default value 42:
    |   Plain annotation: succeeded
    |   Annotated[int, Path()]: failed with AssertionError
    |
    | Falsifying explicit example: test_path_param_default_consistency(
    |     default_value=42,
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 60, in test_path_param_default_consistency
    |     assert case1_succeeded == case2_succeeded, (
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Inconsistent behavior for default value 0:
    |   Plain annotation: succeeded
    |   Annotated[int, Path()]: failed with AssertionError
    |
    | Falsifying explicit example: test_path_param_default_consistency(
    |     default_value=0,
    | )
    +---------------- 3 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 60, in test_path_param_default_consistency
    |     assert case1_succeeded == case2_succeeded, (
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: Inconsistent behavior for default value -1:
    |   Plain annotation: succeeded
    |   Annotated[int, Path()]: failed with AssertionError
    |
    | Falsifying explicit example: test_path_param_default_consistency(
    |     default_value=-1,
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/fastapi_env/lib/python3.13/site-packages')

import inspect
from typing import Annotated
from fastapi.dependencies.utils import analyze_param
from fastapi import Path

# Test Case 1: Plain annotation with default value
print("=" * 60)
print("Test Case 1: Plain annotation with default value")
print("=" * 60)
try:
    details1 = analyze_param(
        param_name="item_id",
        annotation=int,
        value=42,  # Default value
        is_path_param=True
    )
    print(f"✓ Success - Field created")
    print(f"  Field info type: {type(details1.field.field_info)}")
    print(f"  Field default: {details1.field.default}")
    print(f"  Field required: {details1.field.required}")
except AssertionError as e:
    print(f"✗ AssertionError: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {type(e).__name__}: {e}")

print()

# Test Case 2: Annotated with Path() and default value
print("=" * 60)
print("Test Case 2: Annotated with Path() and default value")
print("=" * 60)
try:
    path_info = Path()
    path_info.annotation = int

    details2 = analyze_param(
        param_name="item_id",
        annotation=Annotated[int, path_info],
        value=42,  # Same default value
        is_path_param=True
    )
    print(f"✓ Success - Field created")
    print(f"  Field info type: {type(details2.field.field_info)}")
    print(f"  Field default: {details2.field.default}")
    print(f"  Field required: {details2.field.required}")
except AssertionError as e:
    print(f"✗ AssertionError: {e}")
except Exception as e:
    print(f"✗ Unexpected error: {type(e).__name__}: {e}")

print()
print("=" * 60)
print("Summary:")
print("=" * 60)
print("The same logical operation (creating a path parameter with a default")
print("value of 42) produces DIFFERENT results depending on annotation style:")
print("- Plain annotation: Silently allows the default (but doesn't use it)")
print("- Annotated[int, Path()]: Raises AssertionError")
print()
print("This violates the documented constraint that 'Path parameters")
print("cannot have default values' and creates inconsistent behavior.")
```

<details>

<summary>
Inconsistent enforcement of path parameter default value constraint
</summary>
```
============================================================
Test Case 1: Plain annotation with default value
============================================================
✓ Success - Field created
  Field info type: <class 'fastapi.params.Path'>
  Field default: PydanticUndefined
  Field required: True

============================================================
Test Case 2: Annotated with Path() and default value
============================================================
✗ AssertionError: Path parameters cannot have default values

============================================================
Summary:
============================================================
The same logical operation (creating a path parameter with a default
value of 42) produces DIFFERENT results depending on annotation style:
- Plain annotation: Silently allows the default (but doesn't use it)
- Annotated[int, Path()]: Raises AssertionError

This violates the documented constraint that 'Path parameters
cannot have default values' and creates inconsistent behavior.
```
</details>

## Why This Is A Bug

This is a contract violation that creates an inconsistent API experience. The bug violates several important principles:

1. **Documented constraint violation**: The code explicitly states "Path parameters cannot have default values" in the assertion message (line 395), but only enforces this constraint in one code path.

2. **Inconsistent validation**: The same logical operation—creating a path parameter with a default value—behaves differently based on whether the developer uses `item_id: int = 42` or `item_id: Annotated[int, Path()] = 42`. This violates the principle of least surprise.

3. **Silent failure vs. loud failure**: The plain annotation path silently ignores the default value (converting it to `PydanticUndefined`), while the Annotated path raises an AssertionError. This makes debugging confusing.

4. **Code structure issue**: The validation happens at line 394-396 when `field_info` already exists from Annotated, but is skipped at lines 448-452 when creating a new `field_info` for plain annotations.

5. **Developer confusion**: Developers might think they've successfully set a default value for a path parameter when using plain annotations, but the value is actually ignored, leading to subtle bugs.

## Relevant Context

The bug exists in the `analyze_param` function at `/fastapi/dependencies/utils.py`. The root cause is in the different code paths:

- **Lines 394-396**: When using `Annotated[int, Path()]`, the code checks if it's a path parameter with a default value and raises an AssertionError.

- **Lines 448-452**: When using plain `int` annotation, the code creates a `Path()` field_info without checking if there's a default value. The comment at lines 449-451 mentions that "the same parameter might sometimes be a path parameter and sometimes not", but this doesn't justify the inconsistency.

FastAPI documentation states that path parameters are always required and cannot have default values because they are part of the URL path itself. This makes sense from an HTTP perspective—a path segment either exists or doesn't; there's no concept of a "default" path segment.

The current behavior could lead developers to write code like:

```python
@app.get("/items/{item_id}")
def get_item(item_id: int = 999):  # Developer expects 999 as default
    return {"item_id": item_id}
```

This will silently ignore the default value 999, and the parameter will still be required.

## Proposed Fix

```diff
--- a/fastapi/dependencies/utils.py
+++ b/fastapi/dependencies/utils.py
@@ -445,6 +445,11 @@ def analyze_param(
     # Handle default assignations, neither field_info nor depends was not found in Annotated nor default value
     elif field_info is None and depends is None:
         default_value = value if value is not inspect.Signature.empty else RequiredParam
+        # Validate path parameters cannot have defaults BEFORE creating field_info
+        if is_path_param and value is not inspect.Signature.empty:
+            raise AssertionError(
+                f"Path parameters cannot have default values"
+            )
         if is_path_param:
             # We might check here that `default_value is RequiredParam`, but the fact is that the same
             # parameter might sometimes be a path parameter and sometimes not. See
```