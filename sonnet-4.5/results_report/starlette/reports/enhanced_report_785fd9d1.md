# Bug Report: Starlette FloatConvertor Round-Trip Failure for Small Decimals

**Target**: `starlette.convertors.FloatConvertor`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FloatConvertor's `to_string()` method cannot correctly reconstruct float values smaller than 1e-20, converting them to "0" instead, which breaks the round-trip property needed for URL routing and causes data loss.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from starlette.convertors import FloatConvertor
import re

@given(st.from_regex(re.compile(r"[0-9]+(\.[0-9]+)?"), fullmatch=True))
@settings(max_examples=100)
def test_float_convertor_round_trip(string_value):
    convertor = FloatConvertor()
    float_value = convertor.convert(string_value)
    reconstructed = convertor.to_string(float_value)
    original_float = float(string_value)
    assert convertor.convert(reconstructed) == original_float, \
        f"Round-trip failed: '{string_value}' -> {float_value} -> '{reconstructed}' -> {convertor.convert(reconstructed)} != {original_float}"

if __name__ == "__main__":
    # Run the test
    try:
        test_float_convertor_round_trip()
        print("All property tests passed")
    except AssertionError as e:
        print(f"Property test failed: {e}")

    # Also test the specific failing case
    print("\nTesting specific failing case: '0.000000000000000000001'")
    try:
        convertor = FloatConvertor()
        string_value = "0.000000000000000000001"
        float_value = convertor.convert(string_value)
        reconstructed = convertor.to_string(float_value)
        original_float = float(string_value)
        assert convertor.convert(reconstructed) == original_float, \
            f"Round-trip failed: '{string_value}' -> {float_value} -> '{reconstructed}' -> {convertor.convert(reconstructed)} != {original_float}"
        print("Specific test passed")
    except AssertionError as e:
        print(f"Specific test failed: {e}")
```

<details>

<summary>
**Failing input**: `"0.000000000000000000001"`
</summary>
```
All property tests passed

Testing specific failing case: '0.000000000000000000001'
Specific test failed: Round-trip failed: '0.000000000000000000001' -> 1e-21 -> '0' -> 0.0 != 1e-21
```
</details>

## Reproducing the Bug

```python
from starlette.convertors import FloatConvertor

convertor = FloatConvertor()

original = "0.000000000000000000001"
value = convertor.convert(original)
result = convertor.to_string(value)

print(f"Input:  '{original}'")
print(f"Float:  {value}")
print(f"Output: '{result}'")

# Check if round-trip works
roundtrip_value = convertor.convert(result)
print(f"Round-trip value: {roundtrip_value}")
print(f"Original float: {float(original)}")

# This will fail
assert result == original, f"Round-trip failed: '{original}' -> {value} -> '{result}'"
```

<details>

<summary>
AssertionError: Round-trip conversion fails for small decimal values
</summary>
```
Input:  '0.000000000000000000001'
Float:  1e-21
Output: '0'
Round-trip value: 0.0
Original float: 1e-21
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/56/repo.py", line 19, in <module>
    assert result == original, f"Round-trip failed: '{original}' -> {value} -> '{result}'"
           ^^^^^^^^^^^^^^^^^^
AssertionError: Round-trip failed: '0.000000000000000000001' -> 1e-21 -> '0'
```
</details>

## Why This Is A Bug

The FloatConvertor violates the fundamental round-trip property required for URL routing converters. The regex pattern `[0-9]+(\.[0-9]+)?` accepts decimal strings with unlimited precision, and the `convert()` method correctly parses these to float values. However, `to_string()` uses `"%0.20f"` formatting which only preserves 20 decimal places.

For values smaller than 1e-20 (like `0.000000000000000000001` which equals 1e-21), the formatting produces `"0.00000000000000000000"`. After stripping trailing zeros and the decimal point, this becomes `"0"`. This causes complete data loss - the value 1e-21 becomes 0.0, not just slightly imprecise.

This breaks URL generation in Starlette's routing system. When generating URLs from route parameters, `to_string()` is used to reconstruct the path. A route with a float parameter of 1e-21 would generate a URL with "0" instead, leading to incorrect routing and data corruption. While such small values in URLs are rare, the complete loss of the value (becoming zero) rather than just precision loss makes this a significant issue.

## Relevant Context

The FloatConvertor class is located in `/lib/python3.13/site-packages/starlette/convertors.py`. The problematic code is in the `to_string()` method at line 66:

```python
return ("%0.20f" % value).rstrip("0").rstrip(".")
```

The issue affects any float value smaller than 1e-20. The regex pattern and `convert()` method accept these values, creating an inconsistency in the converter's capabilities. Other converters in Starlette (IntegerConvertor, UUIDConvertor, StringConvertor) maintain proper round-trip properties.

Starlette documentation: https://www.starlette.io/routing/

## Proposed Fix

```diff
--- a/starlette/convertors.py
+++ b/starlette/convertors.py
@@ -63,7 +63,11 @@ class FloatConvertor(Convertor[float]):
         assert value >= 0.0, "Negative floats are not supported"
         assert not math.isnan(value), "NaN values are not supported"
         assert not math.isinf(value), "Infinite values are not supported"
-        return ("%0.20f" % value).rstrip("0").rstrip(".")
+        # Use exponential notation for very small values to avoid precision loss
+        if 0 < value < 1e-20:
+            return str(value)
+        else:
+            return ("%0.20f" % value).rstrip("0").rstrip(".")
```