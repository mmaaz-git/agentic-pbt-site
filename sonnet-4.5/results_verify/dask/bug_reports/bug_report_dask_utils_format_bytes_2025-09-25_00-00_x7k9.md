# Bug Report: dask.utils.format_bytes Length Invariant Violation

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented invariant that "For all values < 2**60, the output is always <= 10 characters." Values >= 1000 PiB produce 11-character output.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_invariant(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"
```

**Failing input**: `n = 1_125_894_277_343_089_729`

## Reproducing the Bug

```python
from dask.utils import format_bytes

n = 1125899906842624000
result = format_bytes(n)

print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)}")
print(f"n < 2**60: {n < 2**60}")

assert len(result) <= 10
```

## Why This Is A Bug

The function's docstring explicitly states: "For all values < 2**60, the output is always <= 10 characters."

However, any value >= 1000 * 2**50 (= 1,125,899,906,842,624,000) produces output like "1000.00 PiB" which has 11 characters, violating this documented guarantee.

## Fix

The issue occurs because the `.2f` format specifier produces 4+ digits when values reach >= 1000 in a given unit. Fix by using conditional formatting to reduce decimal places for large values:

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
+            if value >= 100:
+                return f"{value:.1f} {prefix}B"
+            else:
+                return f"{value:.2f} {prefix}B"
    return f"{n} B"
```