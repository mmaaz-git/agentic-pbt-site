# Bug Report: pydantic.experimental.pipeline _NotIn Constraint

**Target**: `pydantic.experimental.pipeline._apply_constraint` (line 627-633)
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_NotIn` constraint uses bitwise NOT (`operator.__not__`) instead of logical not, causing it to always accept values regardless of whether they're in the exclusion list.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import transform
from typing import Annotated
import pytest

@given(st.integers(), st.lists(st.integers(), min_size=1))
def test_not_in_rejects_excluded_values(value, excluded_values):
    if value not in excluded_values:
        return

    class Model(BaseModel):
        field: Annotated[int, transform(lambda x: x).not_in(excluded_values)]

    with pytest.raises(ValidationError):
        Model(field=value)
```

**Failing input**: Any value that exists in the exclusion list (e.g., `value=2, excluded_values=[1, 2, 3]`)

## Reproducing the Bug

```python
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

class Model(BaseModel):
    field: Annotated[int, transform(lambda x: x).not_in([1, 2, 3])]

result = Model(field=2)
print(f"Accepted value: {result.field}")
```

## Why This Is A Bug

`operator.__not__` is the bitwise NOT operator (`~`), not logical not:
- `~True` returns `-2` (truthy)
- `~False` returns `-1` (truthy)

Since `_check_func` uses `if func(v):` to validate, returning -1 or -2 always passes validation. This means `not_in` never actually rejects any values.

## Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -628,7 +628,7 @@ def _apply_constraint(  # noqa: C901
         values = constraint.values

         def check_not_in(v: Any) -> bool:
-            return operator.__not__(operator.__contains__(values, v))
+            return not operator.__contains__(values, v)

         s = _check_func(check_not_in, f'not in {values}', s)
     else:
```