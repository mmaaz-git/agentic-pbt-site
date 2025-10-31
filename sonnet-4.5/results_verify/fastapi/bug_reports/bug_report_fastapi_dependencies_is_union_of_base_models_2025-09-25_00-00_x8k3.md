# Bug Report: FastAPI Dependencies is_union_of_base_models Logic Error

**Target**: `fastapi.dependencies.utils.is_union_of_base_models`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `is_union_of_base_models` function has a logic error where it would incorrectly return `True` for a Union type with empty arguments, when it should return `False`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from typing import Union, get_args
from fastapi.dependencies.utils import is_union_of_base_models

@given(st.just(Union))
def test_empty_union_should_return_false(union_type):
    """An empty or unparameterized Union should not return True"""
    args = get_args(union_type)

    if len(args) == 0:
        result = is_union_of_base_models(union_type)
        assert result is False, "Empty Union should return False"
```

**Failing input**: Any Union type where `get_args()` returns empty tuple `()`

## Reproducing the Bug

The bug is in the logic flow, not necessarily in real-world usage:

```python
from typing import Union, get_args, get_origin
from fastapi.dependencies.utils import is_union_of_base_models

union_type = Union[str, int]
args = get_args(union_type)

print(f"Args: {args}")
print(f"Length: {len(args)}")
print(f"Result: {is_union_of_base_models(union_type)}")
```

The issue is theoretical but the logic is flawed in utils.py:819-835:

```python
def is_union_of_base_models(field_type: Any) -> bool:
    """Check if field type is a Union where all members are BaseModel subclasses."""
    from fastapi.types import UnionType

    origin = get_origin(field_type)

    if origin is not Union and origin is not UnionType:
        return False

    union_args = get_args(field_type)

    for arg in union_args:        # If union_args is empty (), loop doesn't execute
        if not lenient_issubclass(arg, BaseModel):
            return False

    return True                   # BUG: Returns True for empty union!
```

## Why This Is A Bug

The function's documentation states it "Check if field type is a Union where **all members** are BaseModel subclasses."

If `union_args` is empty:
- The for loop executes 0 times (no iterations)
- No `return False` statement is reached
- Function returns `True`

This violates the function's contract: a Union with zero members cannot have "all members" be BaseModel subclasses. An empty Union should return `False`, not `True`.

While this may not be triggerable with normal Python Union types (since Python simplifies or rejects invalid Unions), the logic is incorrect and could cause issues with:
- Edge cases in type manipulation
- Third-party typing libraries
- Future Python versions with different Union behavior
- Mock/test objects

## Fix

Add an explicit check for empty unions:

```diff
 def is_union_of_base_models(field_type: Any) -> bool:
     """Check if field type is a Union where all members are BaseModel subclasses."""
     from fastapi.types import UnionType

     origin = get_origin(field_type)

     if origin is not Union and origin is not UnionType:
         return False

     union_args = get_args(field_type)
+
+    # Empty union cannot have "all members" be BaseModels
+    if not union_args:
+        return False

     for arg in union_args:
         if not lenient_issubclass(arg, BaseModel):
             return False

     return True
```