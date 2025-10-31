# Bug Report: pydantic.experimental.pipeline Multiple Constraints Apply Validation Twice

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Multiple constraint types (Ge, Lt, Le, Len, MultipleOf) in `_apply_constraint` apply validation checks twice - once in the schema and once via a validator function. This is inconsistent with Gt's implementation and causes unnecessary performance overhead.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform


@given(st.integers(min_value=10, max_value=100))
def test_constraint_efficiency(x):
    class ModelGe(BaseModel):
        value: int = transform(lambda v: v).ge(5)

    class ModelGt(BaseModel):
        value: int = transform(lambda v: v).gt(4)

    class ModelLe(BaseModel):
        value: int = transform(lambda v: v).le(150)

    class ModelLt(BaseModel):
        value: int = transform(lambda v: v).lt(151)

    m_ge = ModelGe(value=x)
    m_gt = ModelGt(value=x)
    m_le = ModelLe(value=x)
    m_lt = ModelLt(value=x)

    assert m_ge.value == m_gt.value == m_le.value == m_lt.value == x


@given(st.text(min_size=5, max_size=20))
def test_len_constraint_double_validation(s):
    class Model(BaseModel):
        value: str = transform(lambda v: v).len(5, 20)

    m = Model(value=s)
    assert m.value == s
    assert 5 <= len(m.value) <= 20
```

**Failing input**: N/A (functionality works but is inefficient)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

with open('/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/experimental/pipeline.py', 'r') as f:
    lines = f.readlines()

print("Gt constraint (lines 448-463) - CORRECT (has else clause):")
print(''.join(lines[447:463]))

print("\nGe constraint (lines 464-478) - BUG (missing else clause):")
print(''.join(lines[463:479]))

print("\nLt constraint (lines 479-493) - BUG (missing else clause):")
print(''.join(lines[478:494]))

print("\nLe constraint (lines 494-508) - BUG (missing else clause):")
print(''.join(lines[493:509]))

print("\nLen constraint (lines 509-533) - BUG (missing else clause):")
print(''.join(lines[508:534]))

print("\nMultipleOf constraint (lines 534-548) - BUG (missing else clause):")
print(''.join(lines[533:549]))
```

## Why This Is A Bug

In `pipeline.py`, the `_apply_constraint` function handles comparison and length constraints. The pattern should be:
1. If the schema type supports native constraint (e.g., int/float/decimal for Ge), apply it to the schema
2. Otherwise, apply a validator function

**Gt constraint implementation (CORRECT)**:
```python
elif isinstance(constraint, annotated_types.Gt):
    gt = constraint.gt
    if s and s['type'] in {'int', 'float', 'decimal'}:
        s = s.copy()
        if s['type'] == 'int' and isinstance(gt, int):
            s['gt'] = gt
        # ... similar for float and decimal
    else:  # ← HAS else clause
        def check_gt(v: Any) -> bool:
            return v > gt
        s = _check_func(check_gt, f'> {gt}', s)
```

**Affected constraints (INCORRECT)**:
- Ge (lines 464-478)
- Lt (lines 479-493)
- Le (lines 494-508)
- Len (lines 509-533)
- MultipleOf (lines 534-548)

All missing the `else:` clause, causing:
```python
# Schema constraint applied when possible
if s and s['type'] in {...}:
    s = s.copy()
    s['constraint_key'] = constraint_value

# Validator ALWAYS applied (should be in else clause)
def check_constraint(v: Any) -> bool:
    return condition

s = _check_func(check_constraint, error_msg, s)
```

This results in:
- **Double validation**: Constraint checked both by schema and by validator function
- **Performance overhead**: Unnecessary function calls during validation
- **Inconsistency**: Different implementation pattern from Gt
- **Code smell**: Violates DRY principle

**Additional issue in Len constraint** (lines 514-521):
Redundant assertion that duplicates the if condition check.

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
             elif s['type'] == 'decimal' and isinstance(lt, Decimal):
                 s['lt'] = lt
-
+        else:
         def check_lt(v: Any) -> bool:
             return v < lt

@@ -503,7 +503,7 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(le, Decimal):
                 s['le'] = le
-
+        else:
         def check_le(v: Any) -> bool:
             return v <= le

@@ -513,14 +513,7 @@ def _apply_constraint(  # noqa: C901
         max_len = constraint.max_length

         if s and s['type'] in {'str', 'list', 'tuple', 'set', 'frozenset', 'dict'}:
-            assert (
-                s['type'] == 'str'
-                or s['type'] == 'list'
-                or s['type'] == 'tuple'
-                or s['type'] == 'set'
-                or s['type'] == 'dict'
-                or s['type'] == 'frozenset'
-            )
             s = s.copy()
             if min_len != 0:
                 s['min_length'] = min_len
             if max_len is not None:
                 s['max_length'] = max_len
-
+        else:
         def check_len(v: Any) -> bool:
             if max_len is not None:
                 return (min_len <= len(v)) and (len(v) <= max_len)
@@ -543,7 +536,7 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(multiple_of, Decimal):
                 s['multiple_of'] = multiple_of
-
+        else:
         def check_multiple_of(v: Any) -> bool:
             return v % multiple_of == 0
```