# Bug Report: pydantic.experimental.pipeline otherwise operator crashes with TypeError

**Target**: `pydantic.experimental.pipeline._check_func` and constraint validation
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `otherwise` operator in pydantic's experimental pipeline API crashes with `TypeError` when constraint checks encounter type mismatches, instead of gracefully falling back to the alternative pipeline. This breaks union type validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic import BaseModel
from typing import Annotated
from pydantic.experimental.pipeline import transform
import pytest


@settings(max_examples=300)
@given(st.one_of(st.integers(), st.text()))
def test_otherwise_with_union_types(value):
    int_pipeline = transform(lambda x: x).ge(0)
    str_pipeline = transform(lambda x: x).str_lower()
    combined_pipeline = int_pipeline.otherwise(str_pipeline)

    class Model(BaseModel):
        x: Annotated[int | str, combined_pipeline]

    result = Model(x=value)

    if isinstance(value, int) and value >= 0:
        assert result.x == value
    elif isinstance(value, str):
        assert result.x == value.lower()
```

**Failing input**:
- String input like `''` or `'HELLO'`: `TypeError: '>=' not supported between instances of 'str' and 'int'`
- Negative integer like `-1`: `TypeError: descriptor 'lower' for 'str' objects doesn't apply to a 'int' object`

## Reproducing the Bug

```python
from pydantic import BaseModel
from typing import Annotated
from pydantic.experimental.pipeline import transform

int_pipeline = transform(lambda x: x).ge(0)
str_pipeline = transform(lambda x: x).str_lower()
combined_pipeline = int_pipeline.otherwise(str_pipeline)

class Model(BaseModel):
    x: Annotated[int | str, combined_pipeline]

Model(x='HELLO')
```

**Output**:
```
TypeError: '>=' not supported between instances of 'str' and 'int'
```

**Second example**:
```python
Model(x=-1)
```

**Output**:
```
TypeError: descriptor 'lower' for 'str' objects doesn't apply to a 'int' object
```

## Why This Is A Bug

The `otherwise` operator is documented (pipeline.py:327) to "return the result of the first chain if it succeeds, and the second chain if it fails." When given a string input with an integer constraint pipeline, the constraint check should fail gracefully (raising a `ValidationError`) and fall back to the string pipeline. Instead, the `TypeError` from comparing `str >= int` propagates up and crashes validation entirely.

Similarly, when given a negative integer that fails the `ge(0)` constraint, the fallback to the string pipeline should recognize the type mismatch and fail appropriately, but instead it crashes trying to call `str.lower()` on an integer.

The root cause is in `_check_func` (pipeline.py:361-374), which doesn't catch `TypeError` exceptions that can occur during predicate evaluation:

```python
def handler(v: Any) -> Any:
    if func(v):  # This can raise TypeError for type mismatches
        return v
    raise ValueError(...)
```

## Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -364,8 +364,12 @@ def _check_func(
     from pydantic_core import core_schema as cs

     def handler(v: Any) -> Any:
-        if func(v):
-            return v
+        try:
+            if func(v):
+                return v
+        except TypeError:
+            # Type mismatch during constraint check - treat as validation failure
+            raise ValueError(f'Expected {predicate_err if isinstance(predicate_err, str) else predicate_err()}')
         raise ValueError(f'Expected {predicate_err if isinstance(predicate_err, str) else predicate_err()}')

     if s is None:
```