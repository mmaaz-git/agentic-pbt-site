# Bug Report: Pydantic Pipeline not_in Constraint Uses Wrong Operator

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_in` constraint in pydantic's experimental pipeline uses `operator.__not__()` (bitwise NOT) instead of logical `not`, causing a TypeError when validating values.

## Property-Based Test

```python
from typing import Annotated
from hypothesis import given, strategies as st
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import validate_as
import pytest


class TestNotInModel(BaseModel):
    value: Annotated[int, validate_as(int).not_in([1, 2, 3])]


@given(st.integers().filter(lambda x: x not in [1, 2, 3]))
def test_not_in_accepts_valid_values(x):
    m = TestNotInModel(value=x)
    assert m.value == x


@given(st.sampled_from([1, 2, 3]))
def test_not_in_rejects_invalid_values(x):
    with pytest.raises(ValidationError):
        TestNotInModel(value=x)
```

**Failing input**: Any value (e.g., `5`)

## Reproducing the Bug

```python
from typing import Annotated
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as


class TestModel(BaseModel):
    value: Annotated[int, validate_as(int).not_in([1, 2, 3])]


try:
    m = TestModel(value=5)
    print(f"Created model: {m}")
except TypeError as e:
    print(f"TypeError: {e}")
```

Output:
```
TypeError: bad operand type for unary ~: 'bool'
```

## Why This Is A Bug

In `pipeline.py` at line 631, the `check_not_in` function uses:

```python
def check_not_in(v: Any) -> bool:
    return operator.__not__(operator.__contains__(values, v))
```

`operator.__not__()` is the bitwise NOT operator (`~`), which only works on integers. When `operator.__contains__` returns a boolean (`True` or `False`), applying bitwise NOT to it raises a TypeError:

```python
>>> operator.__not__(True)
TypeError: bad operand type for unary ~: 'bool'
```

The correct operator for logical negation is the `not` keyword or `operator.not_()` (note the underscore).

## Fix

Replace `operator.__not__` with logical `not`:

```diff
     def check_not_in(v: Any) -> bool:
-        return operator.__not__(operator.__contains__(values, v))
+        return not operator.__contains__(values, v)
```

Or equivalently:

```diff
     def check_not_in(v: Any) -> bool:
-        return operator.__not__(operator.__contains__(values, v))
+        return v not in values
```