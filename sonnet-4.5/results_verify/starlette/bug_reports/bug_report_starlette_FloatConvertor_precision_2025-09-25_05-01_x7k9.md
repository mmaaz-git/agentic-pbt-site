# Bug Report: FloatConvertor Round-Trip Precision Loss

**Target**: `starlette.convertors.FloatConvertor`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FloatConvertor.to_string() loses precision for small floats (< ~1e-20), causing round-trip conversion failures where convert(to_string(x)) != x.

## Property-Based Test

```python
import math
from hypothesis import assume, given, strategies as st
from starlette.convertors import FloatConvertor

@given(st.floats(allow_nan=False, allow_infinity=False, min_value=0.0, max_value=1e10))
def test_float_convertor_round_trip(x):
    convertor = FloatConvertor()
    assume(x >= 0.0)
    assume(not math.isnan(x))
    assume(not math.isinf(x))

    string_repr = convertor.to_string(x)
    result = convertor.convert(string_repr)

    assert math.isclose(result, x, rel_tol=1e-15), f"Round-trip failed: {x} -> {string_repr} -> {result}"
```

**Failing input**: `x = 5.08183882917904e-140` (and any float < ~1e-20)

## Reproducing the Bug

```python
import math
from starlette.convertors import FloatConvertor

convertor = FloatConvertor()
x = 1e-21

string_repr = convertor.to_string(x)
result = convertor.convert(string_repr)

print(f"Original: {x}")
print(f"String: '{string_repr}'")
print(f"Result: {result}")
print(f"Match: {result == x}")
```

Output:
```
Original: 1e-21
String: '0'
Result: 0.0
Match: False
```

## Why This Is A Bug

FloatConvertor is used in URL routing for path parameter conversion (see `routing.py:replace_params()`). The round-trip property (convert ∘ to_string = identity) is critical for URL generation - when you extract a float from a URL path and later reconstruct that URL, the value should be preserved.

The bug occurs because `to_string()` uses `"%0.20f"` formatting which only supports 20 decimal places. Values smaller than ~1e-20 round to "0", violating the round-trip property and causing silent data corruption.

The code explicitly asserts `value >= 0.0` but doesn't document or enforce a minimum non-zero value, suggesting all non-negative floats should work.

## Fix

Use Python's built-in `str()` or format with exponent notation for very small values:

```diff
--- a/starlette/convertors.py
+++ b/starlette/convertors.py
@@ -61,4 +61,8 @@ class FloatConvertor(Convertor[float]):
     def to_string(self, value: float) -> str:
         value = float(value)
         assert value >= 0.0, "Negative floats are not supported"
         assert not math.isnan(value), "NaN values are not supported"
         assert not math.isinf(value), "Infinite values are not supported"
-        return ("%0.20f" % value).rstrip("0").rstrip(".")
+        if value == 0.0:
+            return "0"
+        if value < 1e-10 or value > 1e10:
+            return str(value)
+        return ("%0.20f" % value).rstrip("0").rstrip(".")
```