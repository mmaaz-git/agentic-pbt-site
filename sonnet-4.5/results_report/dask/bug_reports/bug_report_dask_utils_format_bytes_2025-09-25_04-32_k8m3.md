# Bug Report: dask.utils.format_bytes - Length Guarantee Violated

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented guarantee that "For all values < 2**60, the output is always <= 10 characters." Values >= 1000 PiB produce strings with 11 characters.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_claim(n):
    """Property: For all values < 2**60, output is always <= 10 characters"""
    result = format_bytes(n)
    assert len(result) <= 10
```

**Failing input**: `n=1_125_894_277_343_089_729` (approximately 1000 PiB)

## Reproducing the Bug

```python
from dask.utils import format_bytes

pib = 2**50

test_cases = [
    999 * pib,
    1000 * pib,
    1023 * pib,
]

for val in test_cases:
    result = format_bytes(val)
    print(f"{val:20d} -> '{result:12s}' (len={len(result)})")
```

Output:
```
1124774006935781376 -> '999.00 PiB  ' (len=10)
1125899906842624000 -> '1000.00 PiB ' (len=11)
1151795604700004352 -> '1023.00 PiB ' (len=11)
```

The docstring claims: "For all values < 2**60, the output is always <= 10 characters."

But `format_bytes(1000 * 2**50)` returns `'1000.00 PiB'` which has 11 characters.

## Why This Is A Bug

The docstring in `dask/utils.py` (lines 1771-1791) explicitly guarantees:
> "For all values < 2**60, the output is always <= 10 characters."

This contract is violated for values in the range [1000 PiB, 1024 PiB), which is approximately [1.126×10^18, 1.153×10^18). All these values are less than 2^60 but produce strings with 11 characters.

This could affect users who rely on this documented behavior for UI formatting, fixed-width displays, or table layouts.

## Fix

The issue occurs when the coefficient becomes >= 1000. The format string `f"{n / k:.2f} {prefix}B"` produces `"1000.00 PiB"` which is 11 characters.

Solution: Use conditional formatting to reduce decimal places for large coefficients:

```diff
def format_bytes(n: int) -> str:
    """Format bytes as text

    >>> from dask.utils import format_bytes
    >>> format_bytes(1)
    '1 B'
    >>> format_bytes(1234)
    '1.21 kiB'
    >>> format_bytes(12345678)
    '11.77 MiB'
    >>> format_bytes(1234567890)
    '1.15 GiB'
    >>> format_bytes(1234567890000)
    '1.12 TiB'
    >>> format_bytes(1234567890000000)
    '1.10 PiB'

    For all values < 2**60, the output is always <= 10 characters.
    """
    for prefix, k in (
        ("Pi", 2**50),
        ("Ti", 2**40),
        ("Gi", 2**30),
        ("Mi", 2**20),
        ("ki", 2**10),
    ):
        if n >= k * 0.9:
-            return f"{n / k:.2f} {prefix}B"
+            value = n / k
+            if value >= 1000:
+                return f"{value:.0f} {prefix}B"
+            elif value >= 100:
+                return f"{value:.1f} {prefix}B"
+            else:
+                return f"{value:.2f} {prefix}B"
    return f"{n} B"
```

This ensures:
- Values < 100 units: 2 decimal places (e.g., "99.99 PiB" = 10 chars)
- Values 100-999: 1 decimal place (e.g., "999.9 PiB" = 10 chars)
- Values >= 1000: 0 decimal places (e.g., "1000 PiB" = 8 chars)

All outputs remain <= 10 characters for values < 2^60.