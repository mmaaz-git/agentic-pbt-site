# Bug Report: pydantic.experimental.pipeline._apply_constraint Schema Inconsistency for Ge, Lt, Le Constraints

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_apply_constraint` function creates unnecessarily complex schemas for `Ge`, `Lt`, and `Le` constraints by always wrapping them with `function-after` validators, even when the constraint can be set directly in the schema. Only `Gt` follows the correct pattern of conditional validator application.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic.experimental.pipeline import _apply_constraint
from pydantic_core import core_schema as cs
import annotated_types


@given(st.integers(min_value=-1000, max_value=1000))
@settings(max_examples=200)
def test_ge_constraint_schema_structure_matches_gt(value):
    """Test that Ge constraints produce the same schema type as Gt constraints.

    Both Gt and Ge should produce simple schemas when possible (for int/float/decimal types),
    not wrapped function-after schemas.
    """
    int_schema = cs.int_schema()

    ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(value))
    gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(value))

    assert ge_result['type'] == gt_result['type'], \
        f"Ge and Gt should produce same schema type, got {ge_result['type']} vs {gt_result['type']}"


if __name__ == "__main__":
    # Run the test
    test_ge_constraint_schema_structure_matches_gt()
```

<details>

<summary>
**Failing input**: `value=0`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 26, in <module>
    test_ge_constraint_schema_structure_matches_gt()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 8, in test_ge_constraint_schema_structure_matches_gt
    @settings(max_examples=200)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/24/hypo.py", line 20, in test_ge_constraint_schema_structure_matches_gt
    assert ge_result['type'] == gt_result['type'], \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Ge and Gt should produce same schema type, got function-after vs int
Falsifying example: test_ge_constraint_schema_structure_matches_gt(
    value=0,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
from pydantic.experimental.pipeline import _apply_constraint
from pydantic_core import core_schema as cs
import annotated_types

# Create an integer schema
int_schema = cs.int_schema()

# Apply Ge(5) constraint
ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(5))
print(f"Ge(5) result:")
print(f"  Schema type: {ge_result['type']}")
print(f"  Full schema: {ge_result}")
print()

# Apply Gt(5) constraint
gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(5))
print(f"Gt(5) result:")
print(f"  Schema type: {gt_result['type']}")
print(f"  Full schema: {gt_result}")
print()

# Apply Lt(5) constraint
lt_result = _apply_constraint(int_schema.copy(), annotated_types.Lt(5))
print(f"Lt(5) result:")
print(f"  Schema type: {lt_result['type']}")
print(f"  Full schema: {lt_result}")
print()

# Apply Le(5) constraint
le_result = _apply_constraint(int_schema.copy(), annotated_types.Le(5))
print(f"Le(5) result:")
print(f"  Schema type: {le_result['type']}")
print(f"  Full schema: {le_result}")
print()

# Show the inconsistency
print("Comparison:")
print(f"  Gt produces simple schema: {gt_result['type'] == 'int'}")
print(f"  Ge produces wrapped schema: {ge_result['type'] == 'function-after'}")
print(f"  Lt produces wrapped schema: {lt_result['type'] == 'function-after'}")
print(f"  Le produces wrapped schema: {le_result['type'] == 'function-after'}")
```

<details>

<summary>
Demonstrates inconsistent schema generation between Gt and Ge/Lt/Le constraints
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Ge(5) result:
  Schema type: function-after
  Full schema: {'type': 'function-after', 'function': {'type': 'no-info', 'function': <function _check_func.<locals>.handler at 0x73741e160a40>}, 'schema': {'type': 'int', 'ge': 5}}

Gt(5) result:
  Schema type: int
  Full schema: {'type': 'int', 'gt': 5}

Lt(5) result:
  Schema type: function-after
  Full schema: {'type': 'function-after', 'function': {'type': 'no-info', 'function': <function _check_func.<locals>.handler at 0x73741da868e0>}, 'schema': {'type': 'int', 'lt': 5}}

Le(5) result:
  Schema type: function-after
  Full schema: {'type': 'function-after', 'function': {'type': 'no-info', 'function': <function _check_func.<locals>.handler at 0x73741cd3f920>}, 'schema': {'type': 'int', 'le': 5}}

Comparison:
  Gt produces simple schema: True
  Ge produces wrapped schema: True
  Lt produces wrapped schema: True
  Le produces wrapped schema: True
```
</details>

## Why This Is A Bug

This violates the principle of consistency in the codebase. The `_apply_constraint` function in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/experimental/pipeline.py` implements four similar comparison constraints (Gt, Ge, Lt, Le) differently:

1. **Gt constraint** (lines 448-463): Correctly uses an `else` branch to only add a validator function when the constraint cannot be set directly in the schema
2. **Ge, Lt, Le constraints** (lines 464-508): Always add validator functions after setting the constraint, creating unnecessary `function-after` wrappers

The consequence is that Ge, Lt, and Le constraints:
- Create more complex schemas than necessary
- Add runtime overhead with redundant validation (constraint is checked twice - once in schema, once in validator)
- Behave inconsistently with the Gt constraint despite being logically similar operations

The inner schemas correctly have the constraints set (e.g., `'ge': 5`), but they're unnecessarily wrapped with a `function-after` validator that performs the same check, resulting in double validation at runtime.

## Relevant Context

The pydantic.experimental.pipeline module is marked as experimental with a warning that "This module is experimental, its contents are subject to change and deprecation." However, this inconsistency appears to be an unintentional implementation error rather than a deliberate design choice.

Code location: `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/experimental/pipeline.py`

The bug affects three of the four comparison operators:
- `annotated_types.Ge` (greater than or equal)
- `annotated_types.Lt` (less than)
- `annotated_types.Le` (less than or equal)

Only `annotated_types.Gt` (greater than) implements the pattern correctly.

## Proposed Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -471,11 +471,12 @@ def _apply_constraint(  # noqa: C901
                 s['ge'] = ge
             elif s['type'] == 'decimal' and isinstance(ge, Decimal):
                 s['ge'] = ge
+        else:
+            def check_ge(v: Any) -> bool:
+                return v >= ge

-        def check_ge(v: Any) -> bool:
-            return v >= ge
-
-        s = _check_func(check_ge, f'>= {ge}', s)
+            s = _check_func(check_ge, f'>= {ge}', s)
+
     elif isinstance(constraint, annotated_types.Lt):
         lt = constraint.lt
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -486,11 +487,12 @@ def _apply_constraint(  # noqa: C901
                 s['lt'] = lt
             elif s['type'] == 'decimal' and isinstance(lt, Decimal):
                 s['lt'] = lt
+        else:
+            def check_lt(v: Any) -> bool:
+                return v < lt

-        def check_lt(v: Any) -> bool:
-            return v < lt
-
-        s = _check_func(check_lt, f'< {lt}', s)
+            s = _check_func(check_lt, f'< {lt}', s)
+
     elif isinstance(constraint, annotated_types.Le):
         le = constraint.le
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -501,11 +503,11 @@ def _apply_constraint(  # noqa: C901
                 s['le'] = le
             elif s['type'] == 'decimal' and isinstance(le, Decimal):
                 s['le'] = le
+        else:
+            def check_le(v: Any) -> bool:
+                return v <= le

-        def check_le(v: Any) -> bool:
-            return v <= le
-
-        s = _check_func(check_le, f'<= {le}', s)
+            s = _check_func(check_le, f'<= {le}', s)
     elif isinstance(constraint, annotated_types.Len):
         min_len = constraint.min_length
         max_len = constraint.max_length
```