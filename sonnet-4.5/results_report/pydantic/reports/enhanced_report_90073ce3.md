# Bug Report: pydantic.experimental.pipeline Constraints Applied Twice Due to Missing Else Clauses

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Five constraint types (Ge, Lt, Le, Len, MultipleOf) in `_apply_constraint` incorrectly apply constraints twice - once directly to the schema and again as a wrapper validator function - due to missing `else` clauses that would prevent the double application.

## Property-Based Test

```python
import annotated_types
from hypothesis import given, strategies as st, settings
from pydantic_core import core_schema as cs
from pydantic.experimental.pipeline import _apply_constraint


@settings(max_examples=500)
@given(st.integers(min_value=-1000, max_value=1000))
def test_ge_constraint_schema_structure(value):
    """
    Property: Constraint application should be consistent across similar constraint types.

    Gt (correctly) applies constraint once in schema only.
    Ge (bug) applies constraint in schema AND wraps with validator function.
    """
    int_schema = cs.int_schema()

    gt_schema = _apply_constraint(int_schema.copy(), annotated_types.Gt(value))
    ge_schema = _apply_constraint(int_schema.copy(), annotated_types.Ge(value))

    assert gt_schema['type'] == 'int'

    # This assertion fails because Ge incorrectly produces 'function-after'
    # while Gt correctly produces 'int'
    assert ge_schema['type'] == gt_schema['type'], \
        f"Ge produces {ge_schema['type']} but Gt produces {gt_schema['type']}"


if __name__ == "__main__":
    # Run test with hypothesis
    test_ge_constraint_schema_structure()
```

<details>

<summary>
**Failing input**: `value=0` (or any integer value)
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/27
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_ge_constraint_schema_structure FAILED                      [100%]

=================================== FAILURES ===================================
_____________________ test_ge_constraint_schema_structure ______________________
hypo.py:8: in test_ge_constraint_schema_structure
    @given(st.integers(min_value=-1000, max_value=1000))
                   ^^^
hypo.py:25: in test_ge_constraint_schema_structure
    assert ge_schema['type'] == gt_schema['type'], \
E   AssertionError: Ge produces function-after but Gt produces int
E   assert 'function-after' == 'int'
E
E     - int
E     + function-after
E   Falsifying example: test_ge_constraint_schema_structure(
E       value=0,  # or any other generated value
E   )
=============================== warnings summary ===============================
../../../../miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7
  /home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
    warnings.warn(

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
============================ Hypothesis Statistics =============================
hypo.py::test_ge_constraint_schema_structure:

  - during generate phase (0.04 seconds):
    - Typical runtimes: ~ 2-13 ms, of which < 1ms in data generation
    - 0 passing examples, 10 failing examples, 0 invalid examples
    - Found 1 distinct error in this phase

  - during shrink phase (0.24 seconds):
    - Typical runtimes: ~ 2ms, of which < 1ms in data generation
    - 0 passing examples, 95 failing examples, 0 invalid examples
    - Tried 95 shrinks of which 0 were successful

  - Stopped because nothing left to do


=========================== short test summary info ============================
FAILED hypo.py::test_ge_constraint_schema_structure - AssertionError: Ge prod...
```
</details>

## Reproducing the Bug

```python
import annotated_types
from pydantic_core import core_schema as cs
from pydantic.experimental.pipeline import _apply_constraint

# Create a basic integer schema
int_schema = cs.int_schema()

# Apply Gt constraint (correctly implemented)
gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(10))
print("Gt constraint result:")
print(f"  Schema type: {gt_result['type']}")
print(f"  Full schema: {gt_result}")
print()

# Apply Ge constraint (buggy - applies twice)
ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(10))
print("Ge constraint result:")
print(f"  Schema type: {ge_result['type']}")
if ge_result['type'] == 'function-after':
    print(f"  Inner schema: {ge_result['schema']}")
print(f"  Full schema: {ge_result}")
print()

# Apply Lt constraint (buggy - applies twice)
lt_result = _apply_constraint(int_schema.copy(), annotated_types.Lt(100))
print("Lt constraint result:")
print(f"  Schema type: {lt_result['type']}")
if lt_result['type'] == 'function-after':
    print(f"  Inner schema: {lt_result['schema']}")
print(f"  Full schema: {lt_result}")
print()

# Apply Le constraint (buggy - applies twice)
le_result = _apply_constraint(int_schema.copy(), annotated_types.Le(100))
print("Le constraint result:")
print(f"  Schema type: {le_result['type']}")
if le_result['type'] == 'function-after':
    print(f"  Inner schema: {le_result['schema']}")
print(f"  Full schema: {le_result}")
print()

# Apply MultipleOf constraint (buggy - applies twice)
multiple_result = _apply_constraint(int_schema.copy(), annotated_types.MultipleOf(5))
print("MultipleOf constraint result:")
print(f"  Schema type: {multiple_result['type']}")
if multiple_result['type'] == 'function-after':
    print(f"  Inner schema: {multiple_result['schema']}")
print(f"  Full schema: {multiple_result}")
```

<details>

<summary>
Output showing double application of constraints
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Gt constraint result:
  Schema type: int
  Full schema: {'type': 'int', 'gt': 10}

Ge constraint result:
  Schema type: function-after
  Inner schema: {'type': 'int', 'ge': 10}
  Full schema: {'type': 'function-after', 'function': {'type': 'no-info', 'function': <function _check_func.<locals>.handler at 0x77f3ef62d9e0>}, 'schema': {'type': 'int', 'ge': 10}}

Lt constraint result:
  Schema type: function-after
  Inner schema: {'type': 'int', 'lt': 100}
  Full schema: {'type': 'function-after', 'function': {'type': 'no-info', 'function': <function _check_func.<locals>.handler at 0x77f3ef66b600>}, 'schema': {'type': 'int', 'lt': 100}}

Le constraint result:
  Schema type: function-after
  Inner schema: {'type': 'int', 'le': 100}
  Full schema: {'type': 'function-after', 'function': {'type': 'no-info', 'function': <function _check_func.<locals>.handler at 0x77f3ef66b740>}, 'schema': {'type': 'int', 'le': 100}}

MultipleOf constraint result:
  Schema type: function-after
  Inner schema: {'type': 'int', 'multiple_of': 5}
  Full schema: {'type': 'function-after', 'function': {'type': 'no-info', 'function': <function _check_func.<locals>.handler at 0x77f3ef66b880>}, 'schema': {'type': 'int', 'multiple_of': 5}}
```
</details>

## Why This Is A Bug

This violates expected behavior for several important reasons:

1. **Performance Degradation**: Each affected constraint is validated twice - first by pydantic-core's optimized Rust-based schema validation (the `'ge': 10` in the inner schema), and then again by a Python wrapper function. This doubles the validation overhead unnecessarily.

2. **Inconsistent API Behavior**: The `Gt` constraint correctly applies validation only once (producing a simple `{'type': 'int', 'gt': 10}` schema), while the mathematically equivalent `Ge` constraint wraps this in an unnecessary `function-after` schema. Users expect similar constraints to behave consistently.

3. **Code Intent Violation**: The implementation clearly shows optimization intent - the code checks if the schema type supports native constraints (lines 466-473 for Ge) and applies them directly for better performance. However, the missing `else` clause causes the fallback wrapper function to always execute, defeating this optimization.

4. **Documentation Expectations**: While the module is marked as experimental, there is no documentation suggesting that certain constraints should be applied twice or that Ge/Lt/Le should behave differently from Gt. The pattern established by Gt (with its else clause at line 458) represents the intended behavior.

## Relevant Context

The bug affects the following constraint types in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/experimental/pipeline.py`:

- **Ge** (Greater than or equal) - lines 464-478: Missing else before line 475
- **Lt** (Less than) - lines 479-493: Missing else before line 490
- **Le** (Less than or equal) - lines 494-508: Missing else before line 505
- **Len** (Length constraints) - lines 509-533: Missing else before line 528
- **MultipleOf** - lines 534-548: Missing else before line 545

The **Gt** (Greater than) constraint at lines 448-463 demonstrates the correct implementation with an `else` clause at line 458, ensuring the wrapper function is only created when the schema doesn't support native constraints.

This bug impacts any code using pydantic's experimental pipeline module with these constraints, causing unnecessary performance overhead without affecting correctness. While marked experimental, the module is publicly available and this inconsistency could confuse users trying to understand the constraint system.

## Proposed Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -472,10 +472,11 @@ def _apply_constraint(
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
     elif isinstance(constraint, annotated_types.Lt):
         lt = constraint.lt
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -486,10 +487,11 @@ def _apply_constraint(
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
     elif isinstance(constraint, annotated_types.Le):
         le = constraint.le
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -500,10 +502,11 @@ def _apply_constraint(
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
@@ -524,13 +527,14 @@ def _apply_constraint(
                 s['min_length'] = min_len
             if max_len is not None:
                 s['max_length'] = max_len
+        else:
+            def check_len(v: Any) -> bool:
+                if max_len is not None:
+                    return (min_len <= len(v)) and (len(v) <= max_len)
+                return min_len <= len(v)

-        def check_len(v: Any) -> bool:
-            if max_len is not None:
-                return (min_len <= len(v)) and (len(v) <= max_len)
-            return min_len <= len(v)
-
-        s = _check_func(check_len, f'length >= {min_len} and length <= {max_len}', s)
+            s = _check_func(check_len, f'length >= {min_len} and length <= {max_len}', s)
     elif isinstance(constraint, annotated_types.MultipleOf):
         multiple_of = constraint.multiple_of
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -541,10 +545,11 @@ def _apply_constraint(
                 s['multiple_of'] = multiple_of
             elif s['type'] == 'decimal' and isinstance(multiple_of, Decimal):
                 s['multiple_of'] = multiple_of
+        else:
+            def check_multiple_of(v: Any) -> bool:
+                return v % multiple_of == 0

-        def check_multiple_of(v: Any) -> bool:
-            return v % multiple_of == 0
-
-        s = _check_func(check_multiple_of, f'% {multiple_of} == 0', s)
+            s = _check_func(check_multiple_of, f'% {multiple_of} == 0', s)
```