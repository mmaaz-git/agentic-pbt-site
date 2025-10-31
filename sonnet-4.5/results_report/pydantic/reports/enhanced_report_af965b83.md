# Bug Report: pydantic.experimental.pipeline Redundant Validation in Ge/Lt/Le Constraints

**Target**: `pydantic.experimental.pipeline._apply_constraint`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Ge`, `Lt`, and `Le` constraint implementations in `pydantic.experimental.pipeline._apply_constraint` incorrectly apply both schema-level constraints and validator functions, causing redundant double validation. The `Gt` constraint correctly uses an `else` clause to avoid this redundancy.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from hypothesis import given, strategies as st
from pydantic.experimental.pipeline import validate_as
from pydantic import BaseModel
from typing import Annotated

@given(st.integers())
def test_ge_constraint_not_redundant(x):
    """Property: Ge constraint should only validate once"""

    class ModelGe(BaseModel):
        value: Annotated[int, validate_as(int).ge(5)]

    schema = ModelGe.__pydantic_core_schema__['schema']['fields']['value']['schema']

    has_schema_constraint = False
    has_validator_function = False

    # Check for redundant validation
    if isinstance(schema, dict) and 'type' in schema:
        if schema['type'] == 'function-after':
            has_validator_function = True
            inner_schema = schema.get('schema', {})
            if inner_schema.get('type') == 'int' and 'ge' in inner_schema:
                has_schema_constraint = True
        elif schema['type'] == 'int' and 'ge' in schema:
            has_schema_constraint = True

    assert not (has_schema_constraint and has_validator_function), \
        f"Both schema constraint and validator function present - redundant validation! (input: {x})"

@given(st.integers())
def test_lt_constraint_not_redundant(x):
    """Property: Lt constraint should only validate once"""

    class ModelLt(BaseModel):
        value: Annotated[int, validate_as(int).lt(10)]

    schema = ModelLt.__pydantic_core_schema__['schema']['fields']['value']['schema']

    has_schema_constraint = False
    has_validator_function = False

    # Check for redundant validation
    if isinstance(schema, dict) and 'type' in schema:
        if schema['type'] == 'function-after':
            has_validator_function = True
            inner_schema = schema.get('schema', {})
            if inner_schema.get('type') == 'int' and 'lt' in inner_schema:
                has_schema_constraint = True
        elif schema['type'] == 'int' and 'lt' in schema:
            has_schema_constraint = True

    assert not (has_schema_constraint and has_validator_function), \
        f"Both schema constraint and validator function present - redundant validation! (input: {x})"

@given(st.integers())
def test_le_constraint_not_redundant(x):
    """Property: Le constraint should only validate once"""

    class ModelLe(BaseModel):
        value: Annotated[int, validate_as(int).le(10)]

    schema = ModelLe.__pydantic_core_schema__['schema']['fields']['value']['schema']

    has_schema_constraint = False
    has_validator_function = False

    # Check for redundant validation
    if isinstance(schema, dict) and 'type' in schema:
        if schema['type'] == 'function-after':
            has_validator_function = True
            inner_schema = schema.get('schema', {})
            if inner_schema.get('type') == 'int' and 'le' in inner_schema:
                has_schema_constraint = True
        elif schema['type'] == 'int' and 'le' in schema:
            has_schema_constraint = True

    assert not (has_schema_constraint and has_validator_function), \
        f"Both schema constraint and validator function present - redundant validation! (input: {x})"

@given(st.integers())
def test_gt_constraint_correct(x):
    """Property: Gt constraint should only validate once (control test - should pass)"""

    class ModelGt(BaseModel):
        value: Annotated[int, validate_as(int).gt(5)]

    schema = ModelGt.__pydantic_core_schema__['schema']['fields']['value']['schema']

    has_schema_constraint = False
    has_validator_function = False

    # Check for redundant validation
    if isinstance(schema, dict) and 'type' in schema:
        if schema['type'] == 'function-after':
            has_validator_function = True
            inner_schema = schema.get('schema', {})
            if inner_schema.get('type') == 'int' and 'gt' in inner_schema:
                has_schema_constraint = True
        elif schema['type'] == 'int' and 'gt' in schema:
            has_schema_constraint = True

    assert not (has_schema_constraint and has_validator_function), \
        f"Both schema constraint and validator function present - redundant validation! (input: {x})"

# Run the tests
if __name__ == '__main__':
    print("Testing for redundant validation in pydantic.experimental.pipeline constraints")
    print("=" * 80)

    print("\nTesting Gt constraint (should PASS - no redundancy expected):")
    try:
        test_gt_constraint_correct()
        print("✓ PASSED: Gt constraint has no redundant validation")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    print("\nTesting Ge constraint (should FAIL - redundancy expected):")
    try:
        test_ge_constraint_not_redundant()
        print("✓ PASSED: Ge constraint has no redundant validation")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    print("\nTesting Lt constraint (should FAIL - redundancy expected):")
    try:
        test_lt_constraint_not_redundant()
        print("✓ PASSED: Lt constraint has no redundant validation")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")

    print("\nTesting Le constraint (should FAIL - redundancy expected):")
    try:
        test_le_constraint_not_redundant()
        print("✓ PASSED: Le constraint has no redundant validation")
    except AssertionError as e:
        print(f"✗ FAILED: {e}")
```

<details>

<summary>
**Failing input**: `0`
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
Testing for redundant validation in pydantic.experimental.pipeline constraints
================================================================================

Testing Gt constraint (should PASS - no redundancy expected):
✓ PASSED: Gt constraint has no redundant validation

Testing Ge constraint (should FAIL - redundancy expected):
✗ FAILED: Both schema constraint and validator function present - redundant validation! (input: 0)

Testing Lt constraint (should FAIL - redundancy expected):
✗ FAILED: Both schema constraint and validator function present - redundant validation! (input: 0)

Testing Le constraint (should FAIL - redundancy expected):
✗ FAILED: Both schema constraint and validator function present - redundant validation! (input: 0)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from pydantic.experimental.pipeline import validate_as
from pydantic import BaseModel
from typing import Annotated

# Test Gt constraint (correct behavior - no redundancy)
class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(5)]

# Test Ge constraint (bug - has redundant validation)
class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(5)]

# Test Lt constraint (bug - has redundant validation)
class ModelLt(BaseModel):
    value: Annotated[int, validate_as(int).lt(10)]

# Test Le constraint (bug - has redundant validation)
class ModelLe(BaseModel):
    value: Annotated[int, validate_as(int).le(10)]

def print_schema(schema, indent=0):
    """Pretty print schema structure, handling non-JSON serializable parts"""
    spaces = "  " * indent
    if isinstance(schema, dict):
        for key, value in schema.items():
            if key in ['function', 'ref', 'metadata', 'schema_validator', 'model', 'cls']:
                # Skip non-serializable or verbose parts
                print(f"{spaces}{key}: <{type(value).__name__}>")
            elif isinstance(value, (dict, list)):
                print(f"{spaces}{key}:")
                print_schema(value, indent + 1)
            else:
                print(f"{spaces}{key}: {repr(value)}")
    elif isinstance(schema, list):
        for item in schema:
            print(f"{spaces}- ")
            print_schema(item, indent + 1)
    else:
        print(f"{spaces}{repr(schema)}")

# Get the actual field schemas
gt_schema = ModelGt.__pydantic_core_schema__['schema']['fields']['value']['schema']
ge_schema = ModelGe.__pydantic_core_schema__['schema']['fields']['value']['schema']
lt_schema = ModelLt.__pydantic_core_schema__['schema']['fields']['value']['schema']
le_schema = ModelLe.__pydantic_core_schema__['schema']['fields']['value']['schema']

print("=" * 60)
print("Gt constraint (correct - no redundancy):")
print("=" * 60)
print_schema(gt_schema)

print("\n" + "=" * 60)
print("Ge constraint (bug - has redundant validation):")
print("=" * 60)
print_schema(ge_schema)

print("\n" + "=" * 60)
print("Lt constraint (bug - has redundant validation):")
print("=" * 60)
print_schema(lt_schema)

print("\n" + "=" * 60)
print("Le constraint (bug - has redundant validation):")
print("=" * 60)
print_schema(le_schema)

# Test that validation works but is redundant
print("\n" + "=" * 60)
print("Testing validation behavior:")
print("=" * 60)

# Test Gt
try:
    ModelGt(value=6)
    print("✓ Gt: value=6 accepted (> 5)")
except Exception as e:
    print(f"✗ Gt: value=6 rejected: {e}")

try:
    ModelGt(value=5)
    print("✗ Gt: value=5 accepted (should fail)")
except Exception:
    print("✓ Gt: value=5 rejected (not > 5)")

# Test Ge
try:
    ModelGe(value=5)
    print("✓ Ge: value=5 accepted (>= 5)")
except Exception as e:
    print(f"✗ Ge: value=5 rejected: {e}")

try:
    ModelGe(value=4)
    print("✗ Ge: value=4 accepted (should fail)")
except Exception:
    print("✓ Ge: value=4 rejected (not >= 5)")

# Demonstrate the redundancy by checking schema structure
print("\n" + "=" * 60)
print("Schema structure analysis:")
print("=" * 60)

def analyze_schema(schema, name):
    """Check if schema has both schema-level constraint and validator function"""
    has_schema_constraint = False
    has_validator_wrapper = False

    if isinstance(schema, dict) and 'type' in schema:
        if schema['type'] == 'function-after':
            has_validator_wrapper = True
            inner = schema.get('schema', {})
            if inner.get('type') == 'int':
                if any(k in inner for k in ['gt', 'ge', 'lt', 'le']):
                    has_schema_constraint = True
        elif schema['type'] == 'int':
            if any(k in schema for k in ['gt', 'ge', 'lt', 'le']):
                has_schema_constraint = True

    print(f"{name}:")
    print(f"  Schema type: {schema.get('type') if isinstance(schema, dict) else 'unknown'}")
    print(f"  Has schema-level constraint: {has_schema_constraint}")
    print(f"  Has validator function wrapper: {has_validator_wrapper}")
    if has_schema_constraint and has_validator_wrapper:
        print(f"  ⚠️  REDUNDANT VALIDATION DETECTED!")
    else:
        print(f"  ✓ No redundancy")

analyze_schema(gt_schema, "Gt")
analyze_schema(ge_schema, "Ge")
analyze_schema(lt_schema, "Lt")
analyze_schema(le_schema, "Le")

print("\n" + "=" * 60)
print("Summary:")
print("=" * 60)
print("The Gt constraint correctly applies EITHER a schema-level")
print("constraint OR a validator function, but not both.")
print("")
print("The Ge, Lt, and Le constraints incorrectly apply BOTH a")
print("schema-level constraint AND a validator function, causing")
print("redundant validation that degrades performance.")
```

<details>

<summary>
Schema structure comparison showing redundant validation
</summary>
```
/home/npc/miniconda/lib/python3.13/site-packages/pydantic/experimental/__init__.py:7: PydanticExperimentalWarning: This module is experimental, its contents are subject to change and deprecation.
  warnings.warn(
============================================================
Gt constraint (correct - no redundancy):
============================================================
type: 'int'
gt: 5

============================================================
Ge constraint (bug - has redundant validation):
============================================================
function: <dict>
schema:
  type: 'int'
  ge: 5
type: 'function-after'

============================================================
Lt constraint (bug - has redundant validation):
============================================================
function: <dict>
schema:
  type: 'int'
  lt: 10
type: 'function-after'

============================================================
Le constraint (bug - has redundant validation):
============================================================
function: <dict>
schema:
  type: 'int'
  le: 10
type: 'function-after'

============================================================
Testing validation behavior:
============================================================
✓ Gt: value=6 accepted (> 5)
✓ Gt: value=5 rejected (not > 5)
✓ Ge: value=5 accepted (>= 5)
✓ Ge: value=4 rejected (not >= 5)

============================================================
Schema structure analysis:
============================================================
Gt:
  Schema type: int
  Has schema-level constraint: True
  Has validator function wrapper: False
  ✓ No redundancy
Ge:
  Schema type: function-after
  Has schema-level constraint: True
  Has validator function wrapper: True
  ⚠️  REDUNDANT VALIDATION DETECTED!
Lt:
  Schema type: function-after
  Has schema-level constraint: True
  Has validator function wrapper: True
  ⚠️  REDUNDANT VALIDATION DETECTED!
Le:
  Schema type: function-after
  Has schema-level constraint: True
  Has validator function wrapper: True
  ⚠️  REDUNDANT VALIDATION DETECTED!

============================================================
Summary:
============================================================
The Gt constraint correctly applies EITHER a schema-level
constraint OR a validator function, but not both.

The Ge, Lt, and Le constraints incorrectly apply BOTH a
schema-level constraint AND a validator function, causing
redundant validation that degrades performance.
```
</details>

## Why This Is A Bug

This violates expected behavior because the code demonstrates clear internal inconsistency within the same function. The `_apply_constraint` function in `pydantic/experimental/pipeline.py` implements the `Gt` constraint with an `else` clause at line 458 that ensures validator functions are only added when schema-level constraints cannot be applied. However, the `Ge`, `Lt`, and `Le` constraints (lines 464-508) lack these `else` clauses, causing both schema-level constraints AND validator functions to be applied simultaneously.

This results in double validation where:
1. The pydantic-core schema validates the constraint efficiently at the Rust level
2. A Python validator function then redundantly validates the exact same constraint

The redundancy causes unnecessary performance degradation and creates unnecessarily complex schemas. The correct pattern (as demonstrated by `Gt`) is to apply constraints at the schema level when possible OR fall back to validators when necessary, but never both.

## Relevant Context

The bug exists in the experimental pipeline module at `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/experimental/pipeline.py`.

Key observations:
- Lines 448-463: `Gt` constraint correctly uses `else` clause to avoid redundancy
- Lines 464-478: `Ge` constraint missing `else` clause, always adds validator
- Lines 479-493: `Lt` constraint missing `else` clause, always adds validator
- Lines 494-508: `Le` constraint missing `else` clause, always adds validator

The module is marked as experimental with warnings that the API is subject to change. However, this is still a legitimate bug as it represents an internal inconsistency that degrades performance unnecessarily.

Documentation: https://docs.pydantic.dev/latest/api/experimental/#pydantic.experimental.pipeline

## Proposed Fix

```diff
--- a/pydantic/experimental/pipeline.py
+++ b/pydantic/experimental/pipeline.py
@@ -472,10 +472,11 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(ge, Decimal):
                 s['ge'] = ge

-        def check_ge(v: Any) -> bool:
-            return v >= ge
+        else:
+            def check_ge(v: Any) -> bool:
+                return v >= ge

-        s = _check_func(check_ge, f'>= {ge}', s)
+            s = _check_func(check_ge, f'>= {ge}', s)
     elif isinstance(constraint, annotated_types.Lt):
         lt = constraint.lt
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -487,10 +488,11 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(lt, Decimal):
                 s['lt'] = lt

-        def check_lt(v: Any) -> bool:
-            return v < lt
+        else:
+            def check_lt(v: Any) -> bool:
+                return v < lt

-        s = _check_func(check_lt, f'< {lt}', s)
+            s = _check_func(check_lt, f'< {lt}', s)
     elif isinstance(constraint, annotated_types.Le):
         le = constraint.le
         if s and s['type'] in {'int', 'float', 'decimal'}:
@@ -502,10 +504,11 @@ def _apply_constraint(  # noqa: C901
             elif s['type'] == 'decimal' and isinstance(le, Decimal):
                 s['le'] = le

-        def check_le(v: Any) -> bool:
-            return v <= le
+        else:
+            def check_le(v: Any) -> bool:
+                return v <= le

-        s = _check_func(check_le, f'<= {le}', s)
+            s = _check_func(check_le, f'<= {le}', s)
     elif isinstance(constraint, annotated_types.Len):
         min_len = constraint.min_length
         max_len = constraint.max_length
```