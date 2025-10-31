# Bug Report: pandas.io.json ujson Precision Loss

**Target**: `pandas.io.json.ujson_dumps` / `pandas.io.json.ujson_loads`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When serializing large floats with `ujson_dumps`, precision is lost that cannot be recovered with `ujson_loads`. The stdlib `json` module preserves precision correctly, but ujson does not.

## Property-Based Test

```python
from pandas.io.json import ujson_dumps, ujson_loads
from hypothesis import given, settings, strategies as st


@given(st.one_of(
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
    st.booleans(),
    st.none()
))
@settings(max_examples=500)
def test_ujson_roundtrip(obj):
    json_str = ujson_dumps(obj)
    recovered = ujson_loads(json_str)
    assert obj == recovered
```

**Failing input**: `obj=1.0000000000000002e+16`

## Reproducing the Bug

```python
from pandas.io.json import ujson_dumps, ujson_loads
import json

val = 1.0000000000000002e+16
print(f"Original value: {val!r}")

ujson_result = ujson_dumps(val)
print(f"ujson_dumps: {ujson_result}")

recovered_ujson = ujson_loads(ujson_result)
print(f"ujson_loads: {recovered_ujson!r}")
print(f"Precision lost: {val != recovered_ujson}")
print(f"Difference: {val - recovered_ujson}")

stdlib_result = json.dumps(val)
print(f"\nstdlib json.dumps: {stdlib_result}")
recovered_stdlib = json.loads(stdlib_result)
print(f"stdlib json.loads: {recovered_stdlib!r}")
print(f"stdlib preserves: {val == recovered_stdlib}")
```

Output:
```
Original value: 1.0000000000000002e+16
ujson_dumps: 1e+16
ujson_loads: 1e+16
Precision lost: True
Difference: 2.0

stdlib json.dumps: 1.0000000000000002e+16
stdlib json.loads: 1.0000000000000002e+16
stdlib preserves: True
```

## Why This Is A Bug

The `ujson_dumps` function loses precision when serializing large floats. The value `1.0000000000000002e+16` is rounded to `1e+16`, losing 2.0 in the process. This violates the round-trip property that `ujson_loads(ujson_dumps(x)) == x`.

The stdlib `json` module correctly preserves the precision of the float.

## Fix

This is a bug in the ujson C library. The serialization should use sufficient precision to ensure round-trip fidelity. A workaround would be to:
1. Use stdlib json for values that need precision
2. Add a `double_precision` parameter to control the serialization precision
3. Detect precision loss and warn users

The proper fix requires updating the ujson library to serialize floats with full precision by default.