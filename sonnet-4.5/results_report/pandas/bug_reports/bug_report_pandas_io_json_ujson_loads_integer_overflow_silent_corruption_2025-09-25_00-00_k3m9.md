# Bug Report: pandas.io.json.ujson_loads Integer Overflow Silent Corruption

**Target**: `pandas.io.json.ujson_loads`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ujson_loads` silently returns incorrect values (like 0) when deserializing valid JSON integers around -2^64 (-18,446,744,073,709,551,616), causing silent data corruption without any error or warning. This is more dangerous than a crash because it corrupts data without the user knowing.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pandas.io.json import ujson_dumps, ujson_loads

@settings(max_examples=500)
@given(st.dictionaries(st.text(min_size=1), st.integers()))
def test_ujson_dict_roundtrip(d):
    serialized = ujson_dumps(d)
    deserialized = ujson_loads(serialized)
    assert deserialized == d
```

**Failing input**: `{'0': -18446744073709551616}` (where -18446744073709551616 is -2^64)

## Reproducing the Bug

```python
from pandas.io.json import ujson_dumps, ujson_loads

value = -18_446_744_073_709_551_616
serialized = ujson_dumps(value)
deserialized = ujson_loads(serialized)

print(f"Original:     {value}")
print(f"Serialized:   {serialized}")
print(f"Deserialized: {deserialized}")
print(f"Match:        {deserialized == value}")
```

Output:
```
Original:     -18446744073709551616
Serialized:   -18446744073709551616
Deserialized: 0
Match:        False
```

Comparison with stdlib json (which works correctly):
```python
import json

value = -18_446_744_073_709_551_616
serialized = json.dumps(value)
deserialized = json.loads(serialized)
assert deserialized == value
```

## Why This Is A Bug

1. **Silent data corruption**: Unlike the crash at -2^63 - 1, this silently returns an incorrect value (0)
2. **No error indication**: Users have no way to detect this corruption without manually validating every deserialized value
3. **Valid JSON**: The serialized string is valid JSON and can be correctly parsed by stdlib json
4. **Security implications**: Silent data corruption can lead to serious issues in financial, scientific, or other critical applications

This is likely an integer overflow bug where the value wraps around or overflows to 0 in the C implementation.

## Fix

Similar to the underflow bug, this requires either:
1. Using Python's arbitrary precision integers during deserialization (matching stdlib json)
2. Raising an error instead of silently corrupting data

This appears to be an issue in the underlying ujson C library. At minimum, ujson should raise an error for out-of-range integers rather than silently returning incorrect values.