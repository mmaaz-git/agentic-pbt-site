# Bug Report: pandas.io.json.ujson_loads Float Overflow to Infinity

**Target**: `pandas.io.json.ujson_loads`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ujson_loads` silently converts large finite floats near `sys.float_info.max` to positive infinity during round-trip serialization, causing data corruption without any error or warning.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import pandas.io.json as json_module
import math

@given(st.floats(allow_nan=False, allow_infinity=False))
def test_ujson_float_round_trip(f):
    """
    Property: Finite floats should round-trip through ujson
    """
    json_str = json_module.ujson_dumps(f)
    recovered = json_module.ujson_loads(json_str)
    assert math.isclose(f, recovered, rel_tol=1e-9, abs_tol=1e-15) or f == recovered, \
        f"Float round-trip failed: {f} -> {json_str} -> {recovered}"
```

**Failing input**: `1.7976931345e+308` (a large but valid finite float, less than `sys.float_info.max`)

## Reproducing the Bug

```python
import pandas.io.json as json_module

f = 1.7976931345e+308

json_str = json_module.ujson_dumps(f)
recovered = json_module.ujson_loads(json_str)

print(f"Original: {f}")
print(f"JSON: {json_str}")
print(f"Recovered: {recovered}")
print(f"Recovered is inf: {recovered == float('inf')}")
```

Output:
```
Original: 1.7976931345e+308
JSON: 1.797693135e+308
Recovered: inf
Recovered is inf: True
```

## Why This Is A Bug

1. **Silent data corruption**: A finite float value is silently converted to infinity, completely changing the value's meaning and breaking any downstream calculations.

2. **JSON spec violation**: The JSON string `"1.797693135e+308"` represents a finite number. When parsed, it should yield a finite float, not infinity.

3. **Standard library compliance**: Python's standard `json` module correctly preserves this value:
   ```python
   import json
   f = 1.7976931345e+308
   assert f == json.loads(json.dumps(f))  # Works correctly
   ```

4. **Production risk**: In scientific computing, data analysis, or any domain using large numbers, this bug causes silent data corruption. The value `1.8e308` is valid and should not become infinity.

## Fix

The bug is in `ujson_loads` float parsing (likely in the underlying C implementation). When parsing floating-point numbers from JSON strings, the parser appears to be using a representation with insufficient precision or range, causing values to overflow to infinity when they should remain finite.

The fix should ensure that all JSON floating-point strings that represent finite values (i.e., within Python's float range) are parsed as finite floats, not as infinity. The parser should match the behavior of Python's standard library `json.loads()` for float parsing.