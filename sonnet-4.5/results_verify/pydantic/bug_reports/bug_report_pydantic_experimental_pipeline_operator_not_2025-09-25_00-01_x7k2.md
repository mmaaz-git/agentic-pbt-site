# Bug Report: pydantic.experimental.pipeline Invalid use of operator.__not__

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `_NotIn` constraint implementation uses `operator.__not__()` which does not exist in Python's operator module, causing an `AttributeError` when the `not_in()` constraint is used.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform


@given(st.integers(min_value=1, max_value=100))
def test_not_in_constraint_crashes(x):
    class Model(BaseModel):
        value: int = transform(lambda v: v).not_in([5, 10, 15])

    if x not in [5, 10, 15]:
        m = Model(value=x)
        assert m.value == x
```

**Failing input**: Any value that requires validation (e.g., `7`)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from pydantic import BaseModel
from pydantic.experimental.pipeline import transform

class Model(BaseModel):
    value: int = transform(lambda v: v).not_in([5, 10, 15])

try:
    m = Model(value=7)
    print(f"Value: {m.value}")
except AttributeError as e:
    print(f"AttributeError: {e}")
    print("Bug confirmed: operator.__not__ does not exist")
```

Output:
```
AttributeError: module 'operator' has no attribute '__not__'
```

## Why This Is A Bug

In `pipeline.py` line 631, the code uses:
```python
return operator.__not__(operator.__contains__(values, v))
```

However, Python's `operator` module does not have a `__not__` function. The correct functions are:
- `operator.not_(obj)` - Boolean NOT, equivalent to `not obj`
- `operator.invert(obj)` - Bitwise NOT, equivalent to `~obj`

This causes an `AttributeError` whenever a `not_in()` constraint is evaluated, making the feature completely broken.

For comparison, the `_In` constraint (line 624) correctly uses:
```python
return operator.__contains__(values, v)
```

The `_NotIn` constraint should similarly use valid operator module functions.

## Fix

```diff
--- a/pipeline.py
+++ b/pipeline.py
@@ -628,7 +628,7 @@ def _apply_constraint(  # noqa: C901
         values = constraint.values

         def check_not_in(v: Any) -> bool:
-            return operator.__not__(operator.__contains__(values, v))
+            return not operator.__contains__(values, v)

         s = _check_func(check_not_in, f'not in {values}', s)
```

Alternative fix using operator module:
```diff
--- a/pipeline.py
+++ b/pipeline.py
@@ -628,7 +628,7 @@ def _apply_constraint(  # noqa: C901
         values = constraint.values

         def check_not_in(v: Any) -> bool:
-            return operator.__not__(operator.__contains__(values, v))
+            return operator.not_(operator.__contains__(values, v))

         s = _check_func(check_not_in, f'not in {values}', s)
```