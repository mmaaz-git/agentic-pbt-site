# Bug Report: pydantic.experimental.pipeline Constraint Application Redundancy

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_apply_constraint` function applies constraints inconsistently: `Gt` correctly uses an `else` clause to avoid redundant validation, but `Ge`, `Lt`, `Le`, `Len`, and `MultipleOf` always add a redundant function validator wrapper even when the constraint is already embedded in the schema.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.experimental.pipeline import _apply_constraint
import annotated_types
from pydantic_core import core_schema as cs


@given(st.integers(min_value=-1000, max_value=1000))
def test_ge_schema_consistency_with_gt(value):
    int_schema = cs.int_schema()

    gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(value))
    ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(value))

    assert gt_result['type'] == 'int', "Gt should return int schema"
    assert ge_result['type'] == 'int', f"Ge should return int schema like Gt, got '{ge_result['type']}'"
```

**Failing input**: Any integer value (e.g., `5`)

## Reproducing the Bug

```python
from pydantic.experimental.pipeline import _apply_constraint
import annotated_types
from pydantic_core import core_schema as cs

int_schema = cs.int_schema()

gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(5))
ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(5))

print(f"Gt result type: {gt_result['type']}")
print(f"Ge result type: {ge_result['type']}")

assert gt_result['type'] == 'int'
assert ge_result['type'] == 'int'
```

**Expected output:**
```
Gt result type: int
Ge result type: int
```

**Actual output:**
```
Gt result type: int
Ge result type: function
AssertionError
```

## Why This Is A Bug

When applying a `Ge(5)` constraint to an int schema, the code:
1. Correctly adds `s['ge'] = 5` to the schema dict (lines 466-473)
2. **Incorrectly** wraps the schema in a function validator via `_check_func()` (line 478)

This creates redundant validation because the constraint is already enforced at the schema level. The `Gt` constraint correctly avoids this by using an `else` clause (line 458).

**Impact:**
- Performance: Adds unnecessary validation overhead
- Consistency: Different constraints behave differently for no good reason
- Schema introspection: Returns 'function' instead of 'int' schema type
- Affects: `Ge`, `Lt`, `Le`, `Len`, and `MultipleOf` constraints

## Fix

Add the missing `else:` clauses to make `Ge`, `Lt`, `Le`, `Len`, and `MultipleOf` consistent with `Gt`:

```diff
diff --git a/pydantic/experimental/pipeline.py b/pydantic/experimental/pipeline.py
index 1234567..abcdefg 100644
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -471,7 +471,7 @@ def _apply_constraint(
             elif s['type'] == 'decimal' and isinstance(ge, Decimal):
                 s['ge'] = ge
-
+        else:
         def check_ge(v: Any) -> bool:
             return v >= ge

@@ -486,7 +486,7 @@ def _apply_constraint(
             elif s['type'] == 'decimal' and isinstance(lt, Decimal):
                 s['lt'] = lt
-
+        else:
         def check_lt(v: Any) -> bool:
             return v < lt

@@ -501,7 +501,7 @@ def _apply_constraint(
             elif s['type'] == 'decimal' and isinstance(le, Decimal):
                 s['le'] = le
-
+        else:
         def check_le(v: Any) -> bool:
             return v <= le

@@ -525,7 +525,7 @@ def _apply_constraint(
             if max_len is not None:
                 s['max_length'] = max_len
-
+        else:
         def check_len(v: Any) -> bool:
             if max_len is not None:
                 return (min_len <= len(v)) and (len(v) <= max_len)
@@ -542,7 +542,7 @@ def _apply_constraint(
             elif s['type'] == 'decimal' and isinstance(multiple_of, Decimal):
                 s['multiple_of'] = multiple_of
-
+        else:
         def check_multiple_of(v: Any) -> bool:
             return v % multiple_of == 0
```

This ensures that when a constraint can be embedded in the schema dict, it doesn't also add a redundant function wrapper.