# Bug Report: dask.utils.format_bytes Exceeds Documented 10-Character Limit

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented guarantee that "For all values < 2**60, the output is always <= 10 characters" when formatting values >= 1000 PiB, producing 11-character outputs like '1000.00 PiB'.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dask.utils import format_bytes


@given(st.integers(min_value=1, max_value=2**60 - 1))
def test_format_bytes_output_length(n):
    """Test that format_bytes output is always <= 10 characters for values < 2**60"""
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) returned {result!r} with length {len(result)}"


if __name__ == "__main__":
    # Run the test
    test_format_bytes_output_length()
```

<details>

<summary>
**Failing input**: `n=1_125_894_277_343_089_729`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 14, in <module>
    test_format_bytes_output_length()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 6, in test_format_bytes_output_length
    def test_format_bytes_output_length(n):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 9, in test_format_bytes_output_length
    assert len(result) <= 10, f"format_bytes({n}) returned {result!r} with length {len(result)}"
           ^^^^^^^^^^^^^^^^^
AssertionError: format_bytes(1125894277343089729) returned '1000.00 PiB' with length 11
Falsifying example: test_format_bytes_output_length(
    n=1_125_894_277_343_089_729,
)
```
</details>

## Reproducing the Bug

```python
from dask.utils import format_bytes

# Test the boundary case where format_bytes produces 11 characters
n = 1_125_899_906_842_624_000  # This equals 1000 * 2^50
result = format_bytes(n)

print(f"n = {n}")
print(f"n < 2**60 = {n < 2**60}")
print(f"result = {result!r}")
print(f"len(result) = {len(result)}")

# Verify the violation
assert n < 2**60, "Value should be less than 2^60"
assert len(result) == 11, f"Expected length 11, got {len(result)}"
assert result == '1000.00 PiB', f"Expected '1000.00 PiB', got {result!r}"

print("\nContract violation confirmed:")
print(f"  Documentation claims: 'For all values < 2**60, the output is always <= 10 characters'")
print(f"  Actual output: {result!r} has {len(result)} characters")
```

<details>

<summary>
Output shows 11-character result for value < 2^60
</summary>
```
n = 1125899906842624000
n < 2**60 = True
result = '1000.00 PiB'
len(result) = 11

Contract violation confirmed:
  Documentation claims: 'For all values < 2**60, the output is always <= 10 characters'
  Actual output: '1000.00 PiB' has 11 characters
```
</details>

## Why This Is A Bug

The function's docstring at line 1788 in `/dask/utils.py` explicitly states: "For all values < 2**60, the output is always <= 10 characters." This is a clear contract that the function promises to uphold. However, the implementation violates this guarantee for values >= 1000 * 2^50 (approximately 1.125 exabytes).

The violation occurs because the function uses the format string `f"{n / k:.2f} {prefix}B"` which always includes 2 decimal places. When `n / k >= 1000`, the output format becomes:
- 4 digits for the integer part (1000)
- 1 character for the decimal point (.)
- 2 digits for decimal places (00)
- 1 space character
- 3 characters for the unit (PiB)
- Total: 11 characters

Since 1000 * 2^50 = 1,125,899,906,842,624,000 < 2^60 = 1,152,921,504,606,846,976, these 11-character outputs fall within the domain where the documentation guarantees <= 10 characters.

## Relevant Context

The `format_bytes` function is located in `/dask/utils.py` starting at line 1771. The function iterates through binary prefixes (PiB, TiB, GiB, MiB, kiB) and formats the value with the first matching prefix where `n >= k * 0.9`.

The 10-character guarantee is likely intended for fixed-width formatting in tables, logs, or terminal displays. While the edge case only occurs for extremely large values (>= 1000 PiB or ~1.125 exabytes), the documentation uses the word "always" which creates a strict contract with no exceptions.

Documentation link: The function is part of the Dask utilities module, documented at https://docs.dask.org/en/latest/

## Proposed Fix

The simplest fix is to adjust formatting when values reach 1000 or above to maintain the 10-character limit:

```diff
--- a/dask/utils.py
+++ b/dask/utils.py
@@ -1795,7 +1795,11 @@ def format_bytes(n: int) -> str:
         ("ki", 2**10),
     ):
         if n >= k * 0.9:
-            return f"{n / k:.2f} {prefix}B"
+            value = n / k
+            if value >= 1000:
+                # Use 1 decimal place to stay within 10 chars
+                return f"{value:.1f} {prefix}B"
+            return f"{value:.2f} {prefix}B"
     return f"{n} B"
```

This fix preserves the 10-character guarantee by reducing decimal precision to 1 place when the value reaches 1000 or above (e.g., "1000.0 PiB" instead of "1000.00 PiB").