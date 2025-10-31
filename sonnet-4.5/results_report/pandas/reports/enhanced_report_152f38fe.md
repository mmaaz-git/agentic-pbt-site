# Bug Report: pandas.io.json ujson_dumps Precision Loss Causes Float Overflow to Infinity

**Target**: `pandas.io.json.ujson_dumps`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The default precision setting (`double_precision=10`) in `ujson_dumps()` truncates large float values, causing them to overflow to infinity when deserialized, resulting in silent data corruption for valid finite floats near the maximum representable value.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test demonstrating ujson round-trip bug."""

from hypothesis import given, strategies as st, settings
from pandas.io.json import ujson_dumps, ujson_loads
import math


@given(st.floats(allow_nan=False, allow_infinity=False))
@settings(max_examples=500)
def test_ujson_roundtrip_preserves_finiteness(value):
    serialized = ujson_dumps(value)
    deserialized = ujson_loads(serialized)

    if math.isfinite(value):
        assert math.isfinite(deserialized), \
            f"Round-trip should preserve finiteness: {value} -> {serialized} -> {deserialized}"


if __name__ == "__main__":
    # Run the test
    test_ujson_roundtrip_preserves_finiteness()
```

<details>

<summary>
**Failing input**: `1.7976931345e+308`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 22, in <module>
    test_ujson_roundtrip_preserves_finiteness()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 10, in test_ujson_roundtrip_preserves_finiteness
    @settings(max_examples=500)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/17/hypo.py", line 16, in test_ujson_roundtrip_preserves_finiteness
    assert math.isfinite(deserialized), \
           ~~~~~~~~~~~~~^^^^^^^^^^^^^^
AssertionError: Round-trip should preserve finiteness: 1.7976931345e+308 -> 1.797693135e+308 -> inf
Falsifying example: test_ujson_roundtrip_preserves_finiteness(
    value=1.7976931345e+308,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of ujson_dumps precision bug."""

from pandas.io.json import ujson_dumps, ujson_loads
import json
import math

# The problematic value - very close to max float
value = 1.7976931345e+308

print("Testing JSON round-trip with value:", value)
print("Max float value:", 1.7976931348623157e+308)
print()

# Test with standard library json
print("=== Standard Library json ===")
stdlib_serialized = json.dumps(value)
stdlib_result = json.loads(stdlib_serialized)
print(f"Original:    {value}")
print(f"Serialized:  {stdlib_serialized}")
print(f"Deserialized: {stdlib_result}")
print(f"Is finite:    {math.isfinite(stdlib_result)}")
print(f"Matches original: {stdlib_result == value}")
print()

# Test with ujson default precision (10)
print("=== ujson with default precision (10) ===")
ujson_serialized = ujson_dumps(value)
ujson_result = ujson_loads(ujson_serialized)
print(f"Original:    {value}")
print(f"Serialized:  {ujson_serialized}")
print(f"Deserialized: {ujson_result}")
print(f"Is finite:    {math.isfinite(ujson_result)}")
print(f"Matches original: {ujson_result == value}")
print()

# Test with ujson precision 15
print("=== ujson with precision 15 ===")
ujson15_serialized = ujson_dumps(value, double_precision=15)
ujson15_result = ujson_loads(ujson15_serialized)
print(f"Original:    {value}")
print(f"Serialized:  {ujson15_serialized}")
print(f"Deserialized: {ujson15_result}")
print(f"Is finite:    {math.isfinite(ujson15_result)}")
print(f"Matches original: {ujson15_result == value}")
print()

# Demonstrate the bug
print("=== BUG DEMONSTRATION ===")
assert math.isfinite(value), "Original value is finite"
assert math.isfinite(stdlib_result), "stdlib preserves finiteness"
assert not math.isfinite(ujson_result), "ujson default turns finite to infinity!"
print("BUG CONFIRMED: ujson with default precision turns finite value into infinity")
```

<details>

<summary>
Output shows finite value becoming infinity
</summary>
```
Testing JSON round-trip with value: 1.7976931345e+308
Max float value: 1.7976931348623157e+308

=== Standard Library json ===
Original:    1.7976931345e+308
Serialized:  1.7976931345e+308
Deserialized: 1.7976931345e+308
Is finite:    True
Matches original: True

=== ujson with default precision (10) ===
Original:    1.7976931345e+308
Serialized:  1.797693135e+308
Deserialized: inf
Is finite:    False
Matches original: False

=== ujson with precision 15 ===
Original:    1.7976931345e+308
Serialized:  1.7976931345e+308
Deserialized: 1.7976931345e+308
Is finite:    True
Matches original: True

=== BUG DEMONSTRATION ===
BUG CONFIRMED: ujson with default precision turns finite value into infinity
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Data Corruption Without Warning**: Valid finite float values silently become infinity. The value `1.7976931345e+308` is a legitimate finite float (below the maximum of `1.7976931348623157e+308`), yet it gets corrupted to infinity through normal serialization/deserialization.

2. **Round-Trip Property Violation**: A fundamental expectation of JSON serialization is that `loads(dumps(x)) == x` for valid inputs. This property fails for large but valid floats when using default settings.

3. **Inconsistency with Standard Library**: Python's built-in `json` module correctly preserves these values with its default settings. Users migrating from `json` to `pandas.io.json` would experience unexpected data corruption.

4. **Precision Loss at Critical Boundary**: The default `double_precision=10` truncates the mantissa from 11+ significant digits to exactly 10. For the failing value, this changes `1.7976931345e+308` to `1.797693135e+308` in the serialized form. When parsed, this truncated value rounds up past the float maximum, causing overflow.

5. **No Documentation of Risk**: The pandas documentation doesn't warn that the default precision can cause finite values to overflow. The parameter is described as controlling "decimal places" but actually controls significant digits in scientific notation.

## Relevant Context

The issue stems from the ujson library's `double_precision` parameter in `/home/npc/pbt/agentic-pbt/envs/pandas_env/lib/python3.13/site-packages/pandas/io/json/_json.py:145`. This parameter defaults to 10, which is insufficient for edge cases near the float64 boundary.

Key observations:
- Maximum float64 value: `1.7976931348623157e+308`
- Failing value: `1.7976931345e+308` (99.98% of max)
- Serialized with precision 10: `"1.797693135e+308"` (loses the "45" portion)
- When parsed back, `1.797693135e+308` rounds to a value greater than the maximum, resulting in infinity

The ujson library is a C extension imported from `pandas._libs.json`, making this a lower-level issue that affects all pandas JSON operations using the default settings.

Documentation references:
- pandas.DataFrame.to_json documentation: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_json.html
- The `double_precision` parameter accepts values from 0-15, with 15 being the maximum

## Proposed Fix

```diff
--- a/pandas/io/json/_json.py
+++ b/pandas/io/json/_json.py
@@ -142,7 +142,7 @@ def to_json(
     obj: NDFrame,
     orient: str | None = None,
     date_format: str = "epoch",
-    double_precision: int = 10,
+    double_precision: int = 15,
     force_ascii: bool = True,
     date_unit: str = "ms",
     default_handler: Callable[[Any], JSONSerializable] | None = None,
```