# Bug Report: pandas.io.json.ujson_loads Silent Integer Corruption

**Target**: `pandas.io.json.ujson_loads`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ujson_loads` silently corrupts large negative integers (specifically `-2^64`) by converting them to `0`, violating the fundamental round-trip property that JSON serialization should preserve data integrity.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.io.json as json_module

@given(st.integers())
def test_ujson_int_round_trip(n):
    """
    Property: All Python integers should round-trip through ujson
    """
    json_str = json_module.ujson_dumps(n)
    recovered = json_module.ujson_loads(json_str)
    assert n == recovered, f"Integer round-trip failed: {n} -> {json_str} -> {recovered}"
```

**Failing input**: `-18446744073709551616` (which equals `-2^64`)

## Reproducing the Bug

```python
import pandas.io.json as json_module

n = -18446744073709551616

json_str = json_module.ujson_dumps(n)
recovered = json_module.ujson_loads(json_str)

print(f"Original: {n}")
print(f"JSON: {json_str}")
print(f"Recovered: {recovered}")
```

Output:
```
Original: -18446744073709551616
JSON: -18446744073709551616
Recovered: 0
Equal: False
```

## Why This Is A Bug

1. **Silent data corruption**: The function converts `-2^64` to `0` without raising any error or warning, causing complete loss of data integrity.

2. **Inconsistent behavior**: Other out-of-range integers raise explicit errors:
   - `2^64` raises "Value is too big!"
   - `-2^63 - 1` raises "Value is too small"
   - But `-2^64` silently becomes `0` (worst possible behavior)

3. **JSON spec violation**: The JSON specification supports arbitrary-precision integers. Python's standard library `json` module correctly handles `-2^64` and all other arbitrary precision integers.

4. **Production risk**: In data pipelines using pandas, this bug could silently corrupt financial data, timestamps, IDs, or other critical numeric values, with no indication that corruption occurred.

## Fix

The bug is in `ujson_loads` (likely in the underlying C implementation). When parsing integers that underflow below the minimum representable value, it should either:

1. Raise a `ValueError` with a clear message (like it does for `-2^63 - 1`), OR
2. Correctly parse the integer as a Python arbitrary-precision int (like stdlib `json`)

The current behavior of silently converting to `0` is unacceptable and should be fixed immediately. At minimum, it should raise an error rather than silently corrupting data.