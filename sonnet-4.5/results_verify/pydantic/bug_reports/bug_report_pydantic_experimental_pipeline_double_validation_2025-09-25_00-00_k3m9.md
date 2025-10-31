# Bug Report: pydantic.experimental.pipeline Double Validation for Ge/Lt/Le/MultipleOf Constraints

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_apply_constraint` function applies Ge, Lt, Le, and MultipleOf constraints twice - once to the schema and once as a validator function. This is inconsistent with how Gt constraints are handled and causes unnecessary validation overhead.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform


@given(st.integers(min_value=10, max_value=100))
def test_constraint_application_consistency(x):
    class ModelGe(BaseModel):
        value: int = transform(lambda v: v).ge(5)

    class ModelGt(BaseModel):
        value: int = transform(lambda v: v).gt(4)

    m_ge = ModelGe(value=x)
    m_gt = ModelGt(value=x)

    assert m_ge.value == x
    assert m_gt.value == x
```

**Failing input**: N/A (functionality works but is inefficient)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from pydantic.experimental.pipeline import _apply_constraint
import annotated_types

with open('/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/experimental/pipeline.py', 'r') as f:
    lines = f.readlines()

print("Gt constraint (lines 448-463) - HAS else clause:")
print(''.join(lines[447:463]))

print("\nGe constraint (lines 464-478) - MISSING else clause:")
print(''.join(lines[463:479]))

print("\nNotice: Gt has 'else:' before check_gt definition")
print("Notice: Ge does NOT have 'else:', so check_ge always runs")
```

## Why This Is A Bug

In `pipeline.py`, the `_apply_constraint` function handles comparison constraints differently:

1. **Gt constraint (CORRECT)**: Lines 448-463
   - Applies schema constraint `s['gt'] = gt` when possible
   - Uses `else:` clause to only apply validator function when schema constraint can't be used
   - Result: Constraint checked ONCE

2. **Ge/Lt/Le/MultipleOf constraints (BUG)**: Lines 464-478, 479-493, 494-508, 534-548
   - Apply schema constraints when possible (e.g., `s['ge'] = ge`)
   - Missing `else:` clause, so validator function is ALWAYS applied via `_check_func`
   - Result: Constraint checked TWICE (both in schema AND in validator)

This causes:
- Unnecessary performance overhead (double validation)
- Inconsistent implementation across similar constraints
- Violation of DRY principle

## Fix

```diff
--- a/pipeline.py
+++ b/pipeline.py
@@ -471,7 +471,7 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(ge, Decimal):
                 s['ge'] = ge
-
+        else:
         def check_ge(v: Any) -> bool:
             return v >= ge

@@ -487,7 +487,7 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'float' and isinstance(lt, float):
                 s['lt'] = lt
             elif s['type'] == 'decimal' and isinstance(lt, Decimal):
                 s['lt'] = lt
-
+        else:
         def check_lt(v: Any) -> bool:
             return v < lt

@@ -503,7 +503,7 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'float' and isinstance(le, float):
                 s['le'] = le
             elif s['type'] == 'decimal' and isinstance(le, Decimal):
                 s['le'] = le
-
+        else:
         def check_le(v: Any) -> bool:
             return v <= le

@@ -542,7 +542,7 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'float' and isinstance(multiple_of, float):
                 s['multiple_of'] = multiple_of
             elif s['type'] == 'decimal' and isinstance(multiple_of, Decimal):
                 s['multiple_of'] = multiple_of
-
+        else:
         def check_multiple_of(v: Any) -> bool:
             return v % multiple_of == 0
```