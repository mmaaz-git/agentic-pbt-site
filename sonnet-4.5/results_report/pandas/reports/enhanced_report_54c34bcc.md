# Bug Report: pandas.io.json.ujson_loads Integer Overflow Causing Silent Data Corruption

**Target**: `pandas.io.json.ujson_loads`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`ujson_loads` silently corrupts data by returning incorrect values (0 for -2^64, -1 for -2^64-1) when deserializing valid JSON integers near -2^64 boundary, without raising any error or warning, potentially causing undetected data integrity violations in production systems.

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

if __name__ == "__main__":
    test_ujson_dict_roundtrip()
```

<details>

<summary>
**Failing input**: `{'0': -18_446_744_073_709_551_616}` and `{'0': -9_223_372_036_854_775_809}`
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 12, in <module>
  |     test_ujson_dict_roundtrip()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 5, in test_ujson_dict_roundtrip
  |     @given(st.dictionaries(st.text(min_size=1), st.integers()))
  |                    ^^^
  |   File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 9, in test_ujson_dict_roundtrip
    |     assert deserialized == d
    |            ^^^^^^^^^^^^^^^^^
    | AssertionError
    | Falsifying example: test_ujson_dict_roundtrip(
    |     d={'0': -18_446_744_073_709_551_616},
    | )
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/25/hypo.py", line 8, in test_ujson_dict_roundtrip
    |     deserialized = ujson_loads(serialized)
    | ValueError: Value is too small
    | Falsifying example: test_ujson_dict_roundtrip(
    |     d={'0': -9_223_372_036_854_775_809},
    | )
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
from pandas.io.json import ujson_dumps, ujson_loads
import json

# Test value: -2^64 (-18,446,744,073,709,551,616)
value = -18_446_744_073_709_551_616

print("Testing ujson with -2^64:")
print(f"Original value:     {value}")

# Serialize with ujson
serialized = ujson_dumps(value)
print(f"ujson serialized:   {serialized}")

# Deserialize with ujson
deserialized = ujson_loads(serialized)
print(f"ujson deserialized: {deserialized}")

print(f"Values match:       {deserialized == value}")
print()

# Compare with standard library json
print("Testing stdlib json with -2^64:")
std_serialized = json.dumps(value)
std_deserialized = json.loads(std_serialized)
print(f"stdlib json deserialized: {std_deserialized}")
print(f"stdlib values match:      {std_deserialized == value}")
print()

# Test dictionary round-trip (as in the original Hypothesis test)
print("Testing dictionary round-trip:")
test_dict = {'0': value}
print(f"Original dict:      {test_dict}")

dict_serialized = ujson_dumps(test_dict)
print(f"ujson serialized:   {dict_serialized}")

dict_deserialized = ujson_loads(dict_serialized)
print(f"ujson deserialized: {dict_deserialized}")

print(f"Dicts match:        {dict_deserialized == test_dict}")
```

<details>

<summary>
Silent data corruption: ujson returns 0 instead of -18446744073709551616
</summary>
```
Testing ujson with -2^64:
Original value:     -18446744073709551616
ujson serialized:   -18446744073709551616
ujson deserialized: 0
Values match:       False

Testing stdlib json with -2^64:
stdlib json deserialized: -18446744073709551616
stdlib values match:      True

Testing dictionary round-trip:
Original dict:      {'0': -18446744073709551616}
ujson serialized:   {"0":-18446744073709551616}
ujson deserialized: {'0': 0}
Dicts match:        False
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Silent Data Corruption**: Unlike values at -2^63-1 which raise `ValueError: Value is too small`, values at exactly -2^64 silently return 0, and -2^64-1 returns -1. This silent corruption is far more dangerous than an error because it goes undetected.

2. **Violates JSON Specification**: RFC 7159 Section 6 states that while implementations MAY set limits on range and precision, the behavior should be predictable. Silent corruption is never acceptable - parsers should either correctly parse values or raise clear errors.

3. **Inconsistent Behavior Pattern**: The function exhibits three different behaviors:
   - Values in range [-2^63, 2^63-1]: Work correctly
   - Values like -2^63-1, -2^63-2, -2^64+1: Raise `ValueError: Value is too small`
   - Values at -2^64 boundary: Silently corrupt (0 for -2^64, -1 for -2^64-1)

4. **Breaks Round-Trip Guarantee**: ujson successfully serializes -2^64 to the string "-18446744073709551616" but then fails to deserialize it correctly, violating the fundamental round-trip property of serialization.

5. **Incompatible with Python's JSON Module**: Python's standard `json` module correctly handles these values, setting the expectation that JSON parsers in Python should support arbitrary precision integers.

## Relevant Context

Testing revealed the exact boundary behavior:
- `-18446744073709551616` (-2^64): Returns 0 (WRONG)
- `-18446744073709551617` (-2^64 - 1): Returns -1 (WRONG)
- `-18446744073709551615` (-2^64 + 1): Raises ValueError
- `-9223372036854775808` (-2^63): Works correctly
- `-9223372036854775809` (-2^63 - 1): Raises ValueError

The underlying ujson C library appears to have an integer overflow bug where certain values wrap around instead of being properly validated. This is particularly dangerous in:
- Financial applications (monetary values)
- Scientific computing (large negative measurements)
- Cryptographic applications (hash values, keys)
- Data analytics pipelines where data integrity is critical

pandas documentation: https://pandas.pydata.org/docs/reference/api/pandas.io.json.ujson_loads.html
ujson issues: https://github.com/ultrajson/ultrajson/issues

## Proposed Fix

The fix requires modifying the underlying ujson C implementation to properly handle integer boundaries. Two approaches:

1. **Preferred**: Use Python's arbitrary precision integers for values outside int64 range (matching stdlib json behavior)
2. **Minimum**: Consistently raise ValueError for all out-of-range values instead of silent corruption

Since this requires C-level changes in the ujson library, a high-level approach would be:

1. In the ujson decoder, add proper boundary checking before integer conversion
2. For values outside [-2^63, 2^63-1], either:
   - Create a Python long/int object (preferred)
   - Raise a consistent ValueError with clear message
3. Ensure no integer overflow can occur that results in wrapped/corrupted values
4. Add comprehensive tests for all power-of-2 boundaries

As a temporary workaround, pandas users encountering large negative integers should use Python's standard json library:
```python
import json
# Instead of: ujson_loads(data)
# Use: json.loads(data)
```