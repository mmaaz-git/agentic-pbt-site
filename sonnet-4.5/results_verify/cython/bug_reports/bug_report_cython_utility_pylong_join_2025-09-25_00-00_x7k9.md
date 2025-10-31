# Bug Report: Cython.Utility.pylong_join Invalid C Code Generation

**Target**: `Cython.Utility.pylong_join`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `pylong_join` function generates invalid C code when `join_type` parameter is an empty string, producing empty casts `()` which are syntactically invalid in C.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Utility import pylong_join


@given(st.integers(min_value=1, max_value=10), st.text(min_size=0, max_size=5))
@settings(max_examples=1000)
def test_join_type_produces_valid_c_cast(count, join_type):
    result = pylong_join(count, join_type=join_type)

    if join_type == '':
        assert '()' not in result or result.count('()') == 0, \
            f"Empty join_type produces invalid C code with empty casts: {result}"

    cast_pattern = f'({join_type})'
    expected_count = count
    actual_count = result.count(cast_pattern)

    assert actual_count == expected_count, \
        f"Expected {expected_count} occurrences of cast '{cast_pattern}', got {actual_count} in: {result}"
```

**Failing input**: `count=1, join_type=''`

## Reproducing the Bug

```python
from Cython.Utility import pylong_join

result = pylong_join(1, join_type='')
print(result)

result = pylong_join(2, join_type='')
print(result)
```

Output:
```
((()digits[0]))
((((()digits[1]) << PyLong_SHIFT) | ()digits[0]))
```

Both outputs contain invalid C syntax with empty casts `()`.

## Why This Is A Bug

The function is meant to generate valid C code expressions for casting and combining Python long digits. When `join_type` is an empty string, the generated code contains empty casts `()`, which are not valid C syntax and will cause compilation errors. The function should either validate that `join_type` is non-empty, handle empty strings specially, or document this constraint.

## Fix

```diff
def pylong_join(count, digits_ptr='digits', join_type='unsigned long'):
    """
    Generate an unrolled shift-then-or loop over the first 'count' digits.
    Assumes that they fit into 'join_type'.

    (((d[2] << n) | d[1]) << n) | d[0]
    """
+   if not join_type:
+       raise ValueError("join_type must be a non-empty string")
    return ('(' * (count * 2) + ' | '.join(
        "(%s)%s[%d])%s)" % (join_type, digits_ptr, _i, " << PyLong_SHIFT" if _i else '')
        for _i in range(count-1, -1, -1)))
```