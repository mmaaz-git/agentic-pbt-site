# Bug Report: pandas.io.formats.format._trim_zeros_float Unequal Trimming

**Target**: `pandas.io.formats.format._trim_zeros_float`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `_trim_zeros_float` function does not trim trailing zeros equally from all numbers as documented. When given float strings with different numbers of trailing zeros, the function stops trimming as soon as any number no longer ends in '0', resulting in outputs with unequal decimal lengths.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pandas.io.formats.format import _trim_zeros_float


@given(
    num_strs=st.lists(
        st.from_regex(r'^\s*[\+-]?[0-9]+\.[0-9]+0$', fullmatch=True),
        min_size=2,
        max_size=10
    )
)
def test_trim_zeros_float_trims_uniformly(num_strs):
    assume(all('.' in s for s in num_strs))

    result = _trim_zeros_float(num_strs)

    decimal_lengths = []
    for r in result:
        if '.' in r:
            decimal_part = r.split('.')[-1].strip()
            decimal_lengths.append(len(decimal_part))

    if len(decimal_lengths) > 1:
        assert len(set(decimal_lengths)) == 1, f"Unequal decimal lengths: {decimal_lengths}"
```

**Failing input**: `['0.00', '0.0000', '0.00000']`

## Reproducing the Bug

```python
from pandas.io.formats.format import _trim_zeros_float

inputs = ['0.00', '0.0000', '0.00000']
result = _trim_zeros_float(inputs)

print(f"Input:  {inputs}")
print(f"Output: {result}")

decimal_lengths = [len(r.split('.')[1]) for r in result]
print(f"Decimal lengths: {decimal_lengths}")

assert result == ['0.0', '0.00', '0.000']
```

Output:
```
Input:  ['0.00', '0.0000', '0.00000']
Output: ['0.0', '0.00', '0.000']
Decimal lengths: [1, 2, 3]
```

## Why This Is A Bug

The function's docstring states: "Trims the maximum number of trailing zeros equally from all numbers containing decimals". The word "equally" clearly indicates all numbers should have the same number of zeros trimmed, resulting in equal decimal lengths.

However, the current implementation uses a greedy algorithm that trims one zero at a time from all numbers until any number stops ending in '0'. This causes numbers with fewer trailing zeros to finish trimming first, leaving them with shorter decimal parts.

## Fix

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1811,14 +1811,23 @@ def _trim_zeros_float(
         necessary.
     """
     trimmed = str_floats
     number_regex = re.compile(rf"^\s*[\+-]?[0-9]+\{decimal}[0-9]*$")

     def is_number_with_decimal(x) -> bool:
         return re.match(number_regex, x) is not None

-    def should_trim(values: ArrayLike | list[str]) -> bool:
-        """
-        Determine if an array of strings should be trimmed.
-
-        Returns True if all numbers containing decimals (defined by the
-        above regular expression) within the array end in a zero, otherwise
-        returns False.
-        """
-        numbers = [x for x in values if is_number_with_decimal(x)]
-        return len(numbers) > 0 and all(x.endswith("0") for x in numbers)
-
-    while should_trim(trimmed):
-        trimmed = [x[:-1] if is_number_with_decimal(x) else x for x in trimmed]
+    # Find minimum number of trailing zeros among all numbers with decimals
+    numbers = [x for x in trimmed if is_number_with_decimal(x)]
+    if len(numbers) == 0:
+        return list(trimmed)
+
+    min_trailing_zeros = float('inf')
+    for num in numbers:
+        trailing_zeros = len(num) - len(num.rstrip('0'))
+        min_trailing_zeros = min(min_trailing_zeros, trailing_zeros)
+
+    # Trim that many zeros from all numbers
+    if min_trailing_zeros > 0:
+        trimmed = [
+            x[:-min_trailing_zeros] if is_number_with_decimal(x) else x
+            for x in trimmed
+        ]

     # leave one 0 after the decimal points if need be.
     result = [
         x + "0" if is_number_with_decimal(x) and x.endswith(decimal) else x
         for x in trimmed
     ]
     return result
```