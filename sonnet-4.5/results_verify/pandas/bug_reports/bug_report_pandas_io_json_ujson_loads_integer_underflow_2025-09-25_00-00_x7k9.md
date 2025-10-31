# Bug Report: pandas.io.json.ujson_loads Integer Underflow Crash

**Target**: `pandas.io.json.ujson_loads`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ujson_loads` crashes with `ValueError: Value is too small` when deserializing valid JSON integers less than -2^63 (-9,223,372,036,854,775,808), even though `ujson_dumps` successfully serializes them and Python's stdlib `json.loads` handles them correctly.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.json import ujson_dumps, ujson_loads

json_values = st.recursive(
    st.none() | st.booleans() | st.floats(allow_nan=False, allow_infinity=False) | st.integers() | st.text(),
    lambda children: st.lists(children) | st.dictionaries(st.text(), children),
    max_leaves=20
)

@settings(max_examples=1000)
@given(json_values)
def test_ujson_roundtrip(obj):
    serialized = ujson_dumps(obj)
    deserialized = ujson_loads(serialized)
    assert deserialized == obj
```

**Failing input**: `-9223372036854775809` (which is -2^63 - 1)

## Reproducing the Bug

```python
from pandas.io.json import ujson_dumps, ujson_loads

value = -9_223_372_036_854_775_809
serialized = ujson_dumps(value)
print(f"Serialized: {serialized}")

deserialized = ujson_loads(serialized)
```

Output:
```
Serialized: -9223372036854775809
ValueError: Value is too small
```

Comparison with stdlib json (which works correctly):
```python
import json

value = -9_223_372_036_854_775_809
serialized = json.dumps(value)
deserialized = json.loads(serialized)
assert deserialized == value
```

## Why This Is A Bug

1. **Self-inconsistency**: `ujson_dumps` can serialize the integer, but `ujson_loads` cannot deserialize it
2. **Valid JSON**: The serialized string `-9223372036854775809` is valid JSON
3. **Python compatibility**: Python integers support arbitrary precision, and stdlib json handles this correctly
4. **Data loss**: Users cannot round-trip valid Python integers through ujson

The bug occurs because ujson appears to use a fixed-width integer type (likely int64) internally, but only validates the lower bound on deserialization while allowing serialization of any integer.

## Fix

The fix should either:
1. Use Python's arbitrary precision integers during deserialization (matching stdlib json behavior)
2. Or validate integer bounds during serialization to fail fast with a clear error message

This appears to be an issue in the underlying ujson C library that pandas wraps. The limits should match: if `ujson_dumps` accepts an integer, `ujson_loads` should be able to deserialize it.