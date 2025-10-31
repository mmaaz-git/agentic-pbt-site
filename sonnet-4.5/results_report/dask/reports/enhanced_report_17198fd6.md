# Bug Report: dask.utils.format_bytes Violates 10-Character Length Guarantee

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented guarantee that "For all values < 2**60, the output is always <= 10 characters" when formatting values >= 1000 PiB, producing 11-character strings instead.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_claim(n):
    """Property: For all values < 2**60, output is always <= 10 characters"""
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"

if __name__ == "__main__":
    test_format_bytes_length_claim()
```

<details>

<summary>
**Failing input**: `n=1_125_894_277_343_089_729`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 12, in <module>
    test_format_bytes_length_claim()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 5, in test_format_bytes_length_claim
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/8/hypo.py", line 9, in test_format_bytes_length_claim
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"
           ^^^^^^^^^^^^^^^^^
AssertionError: format_bytes(1125894277343089729) = '1000.00 PiB' has length 11 > 10
Falsifying example: test_format_bytes_length_claim(
    n=1_125_894_277_343_089_729,
)
```
</details>

## Reproducing the Bug

```python
from dask.utils import format_bytes

# Test values around 1000 PiB boundary
pib = 2**50

test_cases = [
    999 * pib,     # Should be 10 chars
    1000 * pib,    # Bug: produces 11 chars
    1023 * pib,    # Bug: produces 11 chars
]

print("Testing format_bytes length guarantee:")
print("Docstring claims: 'For all values < 2**60, the output is always <= 10 characters.'")
print(f"2**60 = {2**60}")
print()

for val in test_cases:
    result = format_bytes(val)
    is_valid = len(result) <= 10
    status = "✓" if is_valid else "✗ VIOLATION"
    print(f"Input: {val:22d} (< 2**60: {val < 2**60})")
    print(f"Output: '{result}' (length: {len(result)}) {status}")
    print()
```

<details>

<summary>
Contract violation for values >= 1000 PiB
</summary>
```
Testing format_bytes length guarantee:
Docstring claims: 'For all values < 2**60, the output is always <= 10 characters.'
2**60 = 1152921504606846976

Input:    1124774006935781376 (< 2**60: True)
Output: '999.00 PiB' (length: 10) ✓

Input:    1125899906842624000 (< 2**60: True)
Output: '1000.00 PiB' (length: 11) ✗ VIOLATION

Input:    1151795604700004352 (< 2**60: True)
Output: '1023.00 PiB' (length: 11) ✗ VIOLATION
```
</details>

## Why This Is A Bug

The function's docstring at line 1788 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py` makes an explicit guarantee: "For all values < 2**60, the output is always <= 10 characters." This is a clear, unambiguous contract promise, not merely a description of typical behavior.

The bug occurs specifically when the coefficient (n / k) becomes >= 1000. The current implementation uses `f"{n / k:.2f} {prefix}B"` formatting, which for values like 1000.00 produces:
- "1000.00 PiB" = 11 characters (7 for "1000.00" + 1 space + 3 for "PiB")

This violates the documented guarantee for all values in the range [1000 PiB, 1024 PiB), approximately [1.126×10^18, 1.153×10^18) bytes. While these are extremely large values (over 1 exabyte), they are still less than 2^60 (≈1.153×10^18) and therefore should satisfy the length constraint according to the documentation.

## Relevant Context

The `format_bytes` function is designed to provide human-readable representations of byte quantities using binary prefixes (kiB, MiB, GiB, TiB, PiB). The 10-character limit appears to be an intentional design constraint, likely for fixed-width displays, terminal output formatting, or table layouts where consistent column widths are important.

The function's implementation uses a simple loop through binary prefixes, checking if the value is >= 90% of each unit threshold (e.g., n >= 2^50 * 0.9 for PiB). When it finds a match, it formats with 2 decimal places unconditionally.

Documentation: https://docs.dask.org/en/stable/api.html#dask.utils.format_bytes

## Proposed Fix

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