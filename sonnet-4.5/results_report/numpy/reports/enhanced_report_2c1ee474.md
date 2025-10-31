# Bug Report: pandas.io.formats.format.EngFormatter Silently Loses Precision for Numbers Outside [-24, 24] Exponent Range

**Target**: `pandas.io.formats.format.EngFormatter`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `EngFormatter` class unconditionally clamps exponents to the range [-24, 24] even when `use_eng_prefix=False`, causing silent data loss for very small or very large numbers. Numbers like 1e-50 are incorrectly formatted as "0.0E-24", completely losing their value.

## Property-Based Test

```python
from hypothesis import assume, given, settings, strategies as st
import pandas.io.formats.format as fmt


@given(
    num=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e50, max_value=1e50),
    accuracy=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=500)
def test_engformatter_parse_roundtrip(num, accuracy):
    assume(num != 0)

    formatter = fmt.EngFormatter(accuracy=accuracy, use_eng_prefix=False)
    formatted = formatter(num)

    parsed = float(formatted)
    relative_error = abs(parsed - num) / abs(num) if num != 0 else 0

    max_expected_error = 10 ** (-accuracy + 1)
    assert relative_error < max_expected_error, \
        f"Round-trip error too large: {num} -> '{formatted}' -> {parsed}, error: {relative_error}"

# Run the test
if __name__ == "__main__":
    test_engformatter_parse_roundtrip()
```

<details>

<summary>
**Failing input**: `num=1.401298464324817e-45, accuracy=1`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 25, in <module>
    test_engformatter_parse_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 6, in test_engformatter_parse_roundtrip
    num=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e50, max_value=1e50),
               ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/32/hypo.py", line 20, in test_engformatter_parse_roundtrip
    assert relative_error < max_expected_error, \
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Round-trip error too large: 1.401298464324817e-45 -> ' 0.0E-24' -> 0.0, error: 1.0
Falsifying example: test_engformatter_parse_roundtrip(
    num=1.401298464324817e-45,
    accuracy=1,
)
```
</details>

## Reproducing the Bug

```python
import pandas.io.formats.format as fmt

formatter = fmt.EngFormatter(accuracy=1, use_eng_prefix=False)

# Test with a very small number that's outside the [-24, 24] exponent range
num = 1e-50
formatted = formatter(num)
parsed = float(formatted)

print(f"Original: {num}")
print(f"Formatted: '{formatted}'")
print(f"Parsed back: {parsed}")
print(f"Data loss: {parsed == 0.0 and num != 0.0}")

# Test with the specific failing input from the bug report
num2 = 2.844615157173927e-200
formatted2 = formatter(num2)
parsed2 = float(formatted2)

print(f"\nOriginal: {num2}")
print(f"Formatted: '{formatted2}'")
print(f"Parsed back: {parsed2}")
print(f"Data loss: {parsed2 == 0.0 and num2 != 0.0}")

# Test with a very large number outside the range
num3 = 1e50
formatted3 = formatter(num3)
parsed3 = float(formatted3)

print(f"\nOriginal: {num3}")
print(f"Formatted: '{formatted3}'")
print(f"Parsed back: {parsed3}")
print(f"Expected: 1e50, Got: {parsed3}")
print(f"Data corruption: {parsed3 != num3}")
```

<details>

<summary>
Output showing complete data loss for small numbers
</summary>
```
Original: 1e-50
Formatted: ' 0.0E-24'
Parsed back: 0.0
Data loss: True

Original: 2.844615157173927e-200
Formatted: ' 0.0E-24'
Parsed back: 0.0
Data loss: True

Original: 1e+50
Formatted: ' 100000000000000000000000000.0E+24'
Parsed back: 1e+50
Expected: 1e50, Got: 1e+50
Data corruption: False
```
</details>

## Why This Is A Bug

This bug violates expected behavior because the `EngFormatter` class unconditionally clamps the exponent to match available engineering prefixes, even when the user has explicitly set `use_eng_prefix=False`.

The problematic code is in the `__call__` method at lines 1921-1922 of `/home/npc/miniconda/lib/python3.13/site-packages/pandas/io/formats/format.py`:

```python
pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))  # Clamps to 24
pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))  # Clamps to -24
```

This clamping occurs regardless of the `use_eng_prefix` setting. When `use_eng_prefix=False`, the formatter is supposed to use scientific E notation (like "1.0E-50") which can represent any exponent value, not just those with SI engineering prefixes.

The consequences are severe:
1. **Silent data loss**: Numbers smaller than 1e-24 are rounded to 0, completely losing their value
2. **Incorrect large number representation**: Numbers larger than 1e24 are incorrectly scaled
3. **Broken round-trip property**: A formatted number cannot be parsed back to its original value
4. **Unexpected behavior**: When explicitly choosing not to use engineering prefixes (`use_eng_prefix=False`), users expect standard scientific notation that can handle the full range of floating-point numbers

## Relevant Context

The `ENG_PREFIXES` dictionary defines SI engineering prefixes from yocto (10^-24) to yotta (10^24):
- Smallest: -24 (y, yocto)
- Largest: 24 (Y, yotta)

When `use_eng_prefix=True`, the formatter correctly uses these letter prefixes (e.g., "1.0M" for million). However, when `use_eng_prefix=False`, it should use E notation without any exponent restrictions, as shown in the class docstring example: `'-1.00E-06'`.

The pandas documentation for `set_eng_float_format` shows that E notation is expected to work for any valid float exponent when `use_eng_prefix=False`.

## Proposed Fix

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1918,8 +1918,11 @@ class EngFormatter:
         else:
             pow10 = Decimal(0)

-        pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))
-        pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))
+        if self.use_eng_prefix:
+            # Only clamp when using prefixes, since we need a valid prefix key
+            pow10 = pow10.min(max(self.ENG_PREFIXES.keys()))
+            pow10 = pow10.max(min(self.ENG_PREFIXES.keys()))
+        # else: Allow any exponent when using E notation
         int_pow10 = int(pow10)

         if self.use_eng_prefix:
```