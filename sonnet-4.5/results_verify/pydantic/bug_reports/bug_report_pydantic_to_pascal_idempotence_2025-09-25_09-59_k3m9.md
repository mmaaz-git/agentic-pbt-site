# Bug Report: pydantic.alias_generators.to_pascal Idempotence Violation

**Target**: `pydantic.alias_generators.to_pascal`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `to_pascal` function is not idempotent: applying it twice to the same input produces different results. Specifically, `to_pascal(to_pascal('A_A'))` != `to_pascal('A_A')`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic.alias_generators import to_pascal


@given(st.text(min_size=1))
@settings(max_examples=1000)
def test_to_pascal_idempotent(field_name):
    """to_pascal applied twice should equal to_pascal applied once (idempotence)."""
    once = to_pascal(field_name)
    twice = to_pascal(once)
    assert once == twice
```

**Failing input**: `'A_A'`

## Reproducing the Bug

```python
from pydantic.alias_generators import to_pascal

field = 'A_A'
once = to_pascal(field)
twice = to_pascal(once)

print(f"to_pascal('{field}') = '{once}'")
print(f"to_pascal('{once}') = '{twice}'")
print(f"Expected: '{once}' == '{twice}'")
print(f"Actual: '{once}' != '{twice}'")
```

Output:
```
to_pascal('A_A') = 'AA'
to_pascal('AA') = 'Aa'
Expected: 'AA' == 'Aa'
Actual: 'AA' != 'Aa'
```

## Why This Is A Bug

Idempotence is a fundamental property of transformation functions. When `to_pascal` is used as an `alias_generator`, applying it multiple times (e.g., in chained operations or accidentally) should produce the same result. The current implementation violates this property, which can lead to:

1. Inconsistent field naming when the transformer is applied multiple times
2. Unexpected behavior in serialization/deserialization round-trips
3. Confusion when debugging alias-related issues

A properly implemented PascalCase converter should recognize that 'AA' is already in PascalCase and leave it unchanged, or at minimum, apply the same transformation consistently.

## Fix

The issue appears to be that `to_pascal` treats consecutive uppercase letters differently on subsequent applications. The fix would require `to_pascal` to detect when a string is already in PascalCase format and preserve it, or to normalize the handling of consecutive uppercase letters.