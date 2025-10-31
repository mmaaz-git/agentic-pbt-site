# Bug Report: pydantic.alias_generators.to_snake Idempotence Violation

**Target**: `pydantic.alias_generators.to_snake`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_snake` function is not idempotent: applying it twice to the same input produces different results. Specifically, `to_snake(to_snake('A0'))` != `to_snake('A0')`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic.alias_generators import to_snake


@given(st.text(min_size=1))
@settings(max_examples=1000)
def test_to_snake_idempotent(field_name):
    """to_snake applied twice should equal to_snake applied once (idempotence)."""
    once = to_snake(field_name)
    twice = to_snake(once)
    assert once == twice
```

**Failing input**: `'A0'`

## Reproducing the Bug

```python
from pydantic.alias_generators import to_snake

field = 'A0'
once = to_snake(field)
twice = to_snake(once)

print(f"to_snake('{field}') = '{once}'")
print(f"to_snake('{once}') = '{twice}'")
print(f"Expected: '{once}' == '{twice}'")
print(f"Actual: '{once}' != '{twice}'")
```

Output:
```
to_snake('A0') = 'a0'
to_snake('a0') = 'a_0'
Expected: 'a0' == 'a_0'
Actual: 'a0' != 'a_0'
```

## Why This Is A Bug

Idempotence is a fundamental property of transformation functions. When `to_snake` is used as an `alias_generator`, applying it multiple times should produce the same result. The current implementation violates this property because:

1. First application: `'A0'` → `'a0'` (converts uppercase to lowercase)
2. Second application: `'a0'` → `'a_0'` (inserts underscore before digit)

This inconsistency can lead to:
- Unexpected behavior in serialization/deserialization round-trips
- Inconsistent field naming when the transformer is applied multiple times
- Bugs in code that composes or chains transformations

A properly implemented snake_case converter should produce the same result regardless of how many times it's applied to an already snake_case string.

## Fix

The issue is that `to_snake` doesn't recognize that `'a0'` is already in snake_case format. The function should either:

1. Detect transitions from letters to digits and insert underscores consistently (so `'A0'` should become `'a_0'` on the first application), or
2. Recognize when a string is already in snake_case and preserve it unchanged

The current behavior where the first application produces `'a0'` but the second produces `'a_0'` is inconsistent and violates the expected idempotence property.