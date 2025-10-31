# Bug Report: dask.utils.format_bytes Output Length

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function's docstring claims "For all values < 2**60, the output is always <= 10 characters," but this claim is violated for values >= 100 * 2**50 (approximately 1.13e17).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@settings(max_examples=1000)
@given(st.integers(min_value=0, max_value=2**60 - 1))
def test_format_bytes_output_length(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10"
```

**Failing input**: Any integer >= 100 * 2**50, for example: `112589990684262400`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import format_bytes

test_values = [
    100 * 2**50,
    2**60 - 1,
]

for value in test_values:
    result = format_bytes(value)
    length = len(result)
    print(f"format_bytes({value}) = '{result}' (length: {length})")
    if length > 10:
        print(f"  BUG: Length {length} > 10 for value < 2**60")
```

Output:
```
format_bytes(112589990684262400) = '100.00 PiB' (length: 11)
  BUG: Length 11 > 10 for value < 2**60
format_bytes(1152921504606846975) = '1024.00 PiB' (length: 12)
  BUG: Length 12 > 10 for value < 2**60
```

## Why This Is A Bug

The docstring explicitly claims:
```
For all values < 2**60, the output is always <= 10 characters.
```

This claim is false. The `.2f` format specifier always produces 2 decimal places, so values that format to 100.00 or larger produce more than 10 characters when combined with the unit suffix like " PiB".

The violation occurs when `n / 2**50 >= 100`, which happens at `n >= 100 * 2**50 ≈ 1.13e17`, well below `2**60 ≈ 1.15e18`.

## Fix

The docstring should be corrected to accurately reflect the actual maximum output length, or the implementation should be changed to guarantee the claimed property.

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

-    For all values < 2**60, the output is always <= 10 characters.
+    For most values, the output is <= 10 characters, but values >= 100 * 2**50
+    may produce up to 12 characters (e.g., "1024.00 PiB").
     """
     for prefix, k in (
         ("Pi", 2**50),
         ("Ti", 2**40),
         ("Gi", 2**30),
         ("Mi", 2**20),
         ("ki", 2**10),
     ):
         if n >= k * 0.9:
             return f"{n / k:.2f} {prefix}B"
     return f"{n} B"
```