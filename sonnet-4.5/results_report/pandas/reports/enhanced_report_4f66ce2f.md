# Bug Report: pandas.api.typing.NAType Comparison Type Inconsistency

**Target**: `pandas.api.typing.NAType`
**Severity**: Invalid
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NAType comparisons return `bool` for container types and None instead of `NAType`, but this is intentional design to distinguish data comparisons from structural comparisons.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from pandas.api.typing import NAType


@given(st.lists(st.integers()) | st.dictionaries(st.integers(), st.integers()))
def test_na_equality_consistency_with_containers(container):
    """Test that NA comparisons with containers return NAType, not bool"""
    result = pd.NA == container
    assert isinstance(result, NAType), (
        f"Expected NA == {type(container).__name__} to return NAType, "
        f"got {type(result).__name__}: {result}"
    )


def test_na_equality_consistency_with_none():
    """Test that NA comparisons with None return NAType, not bool"""
    result = pd.NA == None
    assert isinstance(result, NAType), (
        f"Expected NA == None to return NAType, got {type(result).__name__}: {result}"
    )


if __name__ == "__main__":
    # Run the tests
    print("Testing NA equality consistency with None...")
    try:
        test_na_equality_consistency_with_none()
        print("✓ test_na_equality_consistency_with_none passed")
    except AssertionError as e:
        print(f"✗ test_na_equality_consistency_with_none failed: {e}")

    print("\nTesting NA equality consistency with containers...")
    try:
        test_na_equality_consistency_with_containers()
        print("✓ test_na_equality_consistency_with_containers passed")
    except Exception as e:
        print(f"✗ test_na_equality_consistency_with_containers failed: {e}")
```

<details>

<summary>
**Failing input**: `[]`, `{}`, `None`
</summary>
```
Testing NA equality consistency with None...
✗ test_na_equality_consistency_with_none failed: Expected NA == None to return NAType, got bool: False

Testing NA equality consistency with containers...
✗ test_na_equality_consistency_with_containers failed: Expected NA == list to return NAType, got bool: False
```
</details>

## Reproducing the Bug

```python
import pandas as pd

# Test NA comparisons with different types
result_none = pd.NA == None
result_list = pd.NA == []
result_dict = pd.NA == {}
result_tuple = pd.NA == ()
result_set = pd.NA == set()
result_object = pd.NA == object()
result_int = pd.NA == 0
result_str = pd.NA == ""
result_bool = pd.NA == True

print("=== Equality Comparisons with pd.NA ===")
print(f"pd.NA == None: {result_none} (type: {type(result_none).__name__})")
print(f"pd.NA == []: {result_list} (type: {type(result_list).__name__})")
print(f"pd.NA == {{}}: {result_dict} (type: {type(result_dict).__name__})")
print(f"pd.NA == (): {result_tuple} (type: {type(result_tuple).__name__})")
print(f"pd.NA == set(): {result_set} (type: {type(result_set).__name__})")
print(f"pd.NA == object(): {result_object} (type: {type(result_object).__name__})")
print(f"pd.NA == 0: {result_int} (type: {type(result_int).__name__})")
print(f'pd.NA == "": {result_str} (type: {type(result_str).__name__})')
print(f"pd.NA == True: {result_bool} (type: {type(result_bool).__name__})")

print("\n=== Inequality Comparisons with pd.NA ===")
result_ne_none = pd.NA != None
result_ne_list = pd.NA != []
result_ne_int = pd.NA != 0

print(f"pd.NA != None: {result_ne_none} (type: {type(result_ne_none).__name__})")
print(f"pd.NA != []: {result_ne_list} (type: {type(result_ne_list).__name__})")
print(f"pd.NA != 0: {result_ne_int} (type: {type(result_ne_int).__name__})")

print("\n=== Self-Comparison ===")
result_self = pd.NA == pd.NA
print(f"pd.NA == pd.NA: {result_self} (type: {type(result_self).__name__})")
```

<details>

<summary>
Output showing type inconsistency for containers vs data types
</summary>
```
=== Equality Comparisons with pd.NA ===
pd.NA == None: False (type: bool)
pd.NA == []: False (type: bool)
pd.NA == {}: False (type: bool)
pd.NA == (): False (type: bool)
pd.NA == set(): False (type: bool)
pd.NA == object(): False (type: bool)
pd.NA == 0: <NA> (type: NAType)
pd.NA == "": <NA> (type: NAType)
pd.NA == True: <NA> (type: NAType)

=== Inequality Comparisons with pd.NA ===
pd.NA != None: True (type: bool)
pd.NA != []: True (type: bool)
pd.NA != 0: <NA> (type: NAType)

=== Self-Comparison ===
pd.NA == pd.NA: <NA> (type: NAType)
```
</details>

## Why This Is A Bug

This is **NOT a bug** but intentional design. The implementation in `pandas/_libs/missing.pyx` deliberately distinguishes between two categories of comparisons:

1. **Data-like comparisons** (numbers, strings, booleans, arrays, dates): These return `NAType` to preserve three-valued logic and propagate missing value semantics in mathematical operations.

2. **Structural comparisons** (None, containers, generic objects): These return `bool` because they represent type/structural checks rather than data value comparisons.

The `_create_binary_propagating_op` function explicitly handles data types to return NA, while returning `NotImplemented` for unhandled types, triggering Python's fallback to boolean comparisons. This design philosophy ensures mathematical rigor for data operations while maintaining practical usability for structural checks, similar to NULL handling in SQL.

## Relevant Context

Analysis of the pandas source code (`pandas/_libs/missing.pyx`) reveals the implementation:

```python
def _create_binary_propagating_op(name, is_divmod=False):
    def method(self, other):
        # Returns NA for data types
        if (other is C_NA or isinstance(other, (str, bytes))
                or isinstance(other, (numbers.Number, np.bool_))
                or util.is_array(other) and not other.shape):
            return NA
        # ...additional date/time handling...
        # Returns NotImplemented for containers/None/objects
        return NotImplemented
```

When `NotImplemented` is returned, Python's comparison protocol performs identity comparison (`pd.NA is None`), yielding boolean results.

Documentation: https://pandas.pydata.org/docs/user_guide/missing_data.html#NA-propagation

## Proposed Fix

No fix is needed as this is working as designed. The documentation could be clarified:

```diff
- All comparisons with pd.NA return pd.NA
+ Comparisons with pd.NA return pd.NA for data types (numbers, strings,
+ booleans, arrays) to preserve three-valued logic. Comparisons with
+ non-data types (None, containers, objects) return boolean values.
```