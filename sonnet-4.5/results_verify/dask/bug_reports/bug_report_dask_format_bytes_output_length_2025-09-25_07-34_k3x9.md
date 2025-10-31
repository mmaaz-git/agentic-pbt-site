# Bug Report: dask.utils.format_bytes Output Length Violation

**Target**: `dask.utils.format_bytes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented invariant that "For all values < 2**60, the output is always <= 10 characters" when formatting values >= 1000 PiB.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_output_length_invariant(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"
```

**Failing input**: `1_125_894_277_343_089_729`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from dask.utils import format_bytes

n = 1_125_894_277_343_089_729
result = format_bytes(n)

print(f"format_bytes({n}) = '{result}'")
print(f"Length: {len(result)} characters")
print(f"Expected: <= 10 characters")
print(f"Violates documented invariant: {len(result) > 10}")
```

Output:
```
format_bytes(1125894277343089729) = '1000.00 PiB'
Length: 11 characters
Expected: <= 10 characters
Violates documented invariant: True
```

## Why This Is A Bug

The function's docstring explicitly states: "For all values < 2**60, the output is always <= 10 characters." This is a documented contract that users may rely on for UI layout, formatting, or buffer allocation.

The bug occurs because values >= 1000 PiB produce output like "1000.00 PiB" (11 characters) instead of staying within the 10-character limit.

The issue is in the formatting logic at dask/utils.py:1798:
```python
return f"{n / k:.2f} {prefix}B"
```

When n/k >= 1000, the format string produces outputs like "1000.00" (7 chars) + " " (1 char) + "PiB" (3 chars) = 11 characters total.

## Fix

The fix should ensure values >= 1000 in a unit are displayed in the next larger unit, or adjust the precision to maintain the 10-character limit:

```diff
def format_bytes(n: int) -> str:
    for prefix, k in (
        ("Pi", 2**50),
        ("Ti", 2**40),
        ("Gi", 2**30),
        ("Mi", 2**20),
        ("ki", 2**10),
    ):
-       if n >= k * 0.9:
+       if n >= k * 0.9 and n < k * 1000:
            return f"{n / k:.2f} {prefix}B"
+   if n >= 2**60:
+       return f"{n / 2**50:.0f} PiB"
    return f"{n} B"
```

This fix ensures that values >= 1000 PiB (which would exceed 10 characters with 2 decimal places) are formatted without decimal places, staying within the 10-character limit.