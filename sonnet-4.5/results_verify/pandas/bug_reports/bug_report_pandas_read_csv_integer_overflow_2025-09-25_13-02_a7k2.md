# Bug Report: pandas.io.parsers read_csv Integer Overflow Silent Data Corruption

**Target**: `pandas.io.parsers.read_csv`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When reading CSV data with `dtype` specified, values that exceed the dtype's range silently overflow/wraparound instead of raising an error. For example, reading `2147483648` (2^31) with `dtype='int32'` produces `-2147483648` (int32 min) due to integer wraparound.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas as pd
from io import StringIO

@given(
    st.lists(
        st.tuples(st.integers(), st.integers()),
        min_size=1,
        max_size=50
    )
)
def test_dtype_specification_preserved(rows):
    csv_string = "a,b\n" + "\n".join(f"{a},{b}" for a, b in rows)
    result = pd.read_csv(StringIO(csv_string), dtype={'a': 'int64', 'b': 'int32'})

    assert result['a'].dtype == np.int64
    assert result['b'].dtype == np.int32
```

**Failing input**: `rows=[(0, 2147483648)]` where `2147483648 > int32_max`

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO

csv = "value\n2147483648"
result = pd.read_csv(StringIO(csv), dtype={'value': 'int32'})

print(f"Input value:  2147483648")
print(f"int32 max:    2147483647")
print(f"Result value: {result['value'].iloc[0]}")
```

Output:
```
Input value:  2147483648
int32 max:    2147483647
Result value: -2147483648
```

The value silently wraps around from `2147483648` to `-2147483648`.

## Why This Is A Bug

1. **Silent data corruption**: Integer overflow occurs without warning or error
2. **Violates user expectations**: Specifying a dtype should either parse correctly or raise an error
3. **Classic security issue**: Integer overflow bugs are a well-known class of vulnerabilities
4. **Breaks data integrity**: Financial, scientific, or ID data could be silently corrupted
5. **Inconsistent with pandas behavior**: Pandas typically raises errors for invalid conversions

Expected behavior: Should raise `OverflowError` or `ValueError: value 2147483648 out of range for int32`.

## Fix

The C parser should check for overflow when converting to fixed-width integer types. Suggested fix in `c_parser_wrapper.py` or the underlying C parsing code:

1. Before assignment to int32/int64, check if the parsed value is within bounds
2. If out of bounds, raise `OverflowError` with a descriptive message
3. This check should occur in `_try_int64()` and similar conversion functions

Example fix pattern (in C or Cython):
```c
if (value > INT32_MAX || value < INT32_MIN) {
    raise OverflowError("value out of range for int32");
}
```