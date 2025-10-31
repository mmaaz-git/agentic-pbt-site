# Bug Report: FloatConvertor Round-Trip Failure

**Target**: `starlette.convertors.FloatConvertor`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

FloatConvertor's `to_string()` method cannot reconstruct small decimal values that its own regex accepts, breaking the round-trip property required for URL routing.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from starlette.convertors import FloatConvertor
import re

@given(st.from_regex(re.compile(r"[0-9]+(\.[0-9]+)?"), fullmatch=True))
def test_float_convertor_round_trip(string_value):
    convertor = FloatConvertor()
    float_value = convertor.convert(string_value)
    reconstructed = convertor.to_string(float_value)
    original_float = float(string_value)
    assert convertor.convert(reconstructed) == original_float
```

**Failing input**: `"0.000000000000000000001"`

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

assert float(result) == float(original)
```

Output:
```
Input:  '0.000000000000000000001'
Float:  1e-21
Output: '0'
AssertionError: 0.0 != 1e-21
```

## Why This Is A Bug

The FloatConvertor regex `[0-9]+(\.[0-9]+)?` accepts decimal strings with any number of digits, but `to_string()` uses `%0.20f` formatting which only preserves 20 decimal places. Values requiring >20 decimal places (like `0.000000000000000000001` = 1e-21) are formatted as `0.00000000000000000000`, which becomes `"0"` after stripping trailing zeros.

This violates the round-trip property needed for URL routing: `convert(to_string(convert(s)))` should equal `convert(s)`. When routing uses `to_string()` to generate URLs, small floats get corrupted to `"0"`, causing URL path parameter mismatch.

## Fix

The cleanest fix is to restrict the regex to only accept values that `to_string()` can faithfully represent. Since `%0.20f` can handle up to 20 decimal places, limit the regex accordingly:

```diff
--- a/starlette/convertors.py
+++ b/starlette/convertors.py
@@ -54,7 +54,7 @@ class IntegerConvertor(Convertor[int]):


 class FloatConvertor(Convertor[float]):
-    regex = r"[0-9]+(\.[0-9]+)?"
+    regex = r"[0-9]+(\.[0-9]{1,20})?"

     def convert(self, value: str) -> float:
         return float(value)
```