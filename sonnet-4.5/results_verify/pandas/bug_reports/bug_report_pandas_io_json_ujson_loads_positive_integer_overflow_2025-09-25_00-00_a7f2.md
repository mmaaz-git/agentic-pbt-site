# Bug Report: pandas.io.json.ujson_loads Positive Integer Overflow Crash

**Target**: `pandas.io.json.ujson_loads`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`ujson_loads` crashes with `ValueError: Value is too big!` when deserializing valid JSON integers greater than or equal to 2^64 (18,446,744,073,709,551,616), even though `ujson_dumps` successfully serializes them and Python's stdlib `json.loads` handles them correctly.

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

**Failing input**: `18446744073709551616` (which is 2^64)

## Reproducing the Bug

```python
from pandas.io.json import ujson_dumps, ujson_loads

value = 2**64
serialized = ujson_dumps(value)
print(f"Serialized: {serialized}")

deserialized = ujson_loads(serialized)
```

Output:
```
Serialized: 18446744073709551616
ValueError: Value is too big!
```

Comparison with stdlib json (which works correctly):
```python
import json

value = 2**64
serialized = json.dumps(value)
deserialized = json.loads(serialized)
assert deserialized == value
```

Testing the boundary:
```python
from pandas.io.json import ujson_dumps, ujson_loads

ujson_loads(ujson_dumps(2**64 - 1))
ujson_loads(ujson_dumps(2**64))
```

## Why This Is A Bug

1. **Self-inconsistency**: `ujson_dumps` can serialize the integer, but `ujson_loads` cannot deserialize it
2. **Valid JSON**: The serialized string `18446744073709551616` is valid JSON
3. **Python compatibility**: Python integers support arbitrary precision, and stdlib json handles this correctly
4. **Asymmetric bounds**: ujson accepts negative integers down to -2^63 but only positive integers up to 2^64 - 1, suggesting it uses int64 for negative and uint64 for positive values

The ujson library appears to use fixed-width 64-bit integers internally (int64 for negative, uint64 for positive), with range [-2^63, 2^64-1]. While this covers a wide range, it's incompatible with:
- Python's arbitrary precision integers
- The JSON specification (which doesn't specify integer size limits)
- The stdlib json library behavior

## Fix

The fix should either:
1. Use Python's arbitrary precision integers during deserialization (matching stdlib json behavior)
2. Or validate integer bounds during serialization to fail fast with a clear error message

This appears to be an issue in the underlying ujson C library that pandas wraps. The limits should match: if `ujson_dumps` accepts an integer, `ujson_loads` should be able to deserialize it.