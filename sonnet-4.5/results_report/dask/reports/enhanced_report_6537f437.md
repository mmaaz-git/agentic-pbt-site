# Bug Report: dask.utils.format_bytes Violates Documented 10-Character Length Guarantee

**Target**: `dask.utils.format_bytes`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented guarantee that "For all values < 2**60, the output is always <= 10 characters" when formatting values >= 1000 PiB (approximately 1.126e18 bytes).

## Property-Based Test

```python
#!/usr/bin/env python3
from hypothesis import given, settings, strategies as st
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_length_claim(n):
    """
    Test the documented claim: "For all values < 2**60, the output is always <= 10 characters."
    """
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10"

if __name__ == "__main__":
    test_format_bytes_length_claim()
```

<details>

<summary>
**Failing input**: `n=1_125_894_277_343_089_729`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 15, in <module>
    test_format_bytes_length_claim()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 6, in test_format_bytes_length_claim
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/46/hypo.py", line 12, in test_format_bytes_length_claim
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)}, expected <= 10"
           ^^^^^^^^^^^^^^^^^
AssertionError: format_bytes(1125894277343089729) = '1000.00 PiB' has length 11, expected <= 10
Falsifying example: test_format_bytes_length_claim(
    n=1_125_894_277_343_089_729,
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
from dask.utils import format_bytes

# Test the specific failing case
n = 1_125_899_906_842_624_000
result = format_bytes(n)
print(f"Input: {n}")
print(f"Result: '{result}'")
print(f"Length: {len(result)} characters")
print()

# Test boundary cases
test_cases = [
    (999 * 2**50, "999 * 2**50"),
    (1000 * 2**50, "1000 * 2**50"),
    (1023 * 2**50, "1023 * 2**50"),
]

print("Boundary cases:")
for value, description in test_cases:
    result = format_bytes(value)
    print(f"  {description} = '{result}' ({len(result)} chars)")

# Verify the documented guarantee
print()
print("Checking documented guarantee: 'For all values < 2**60, the output is always <= 10 characters'")
print(f"Is {n} < 2**60? {n < 2**60}")
print(f"Is output <= 10 characters? {len(format_bytes(n)) <= 10}")

# Trigger the assertion error
assert len(format_bytes(n)) <= 10, f"Expected <= 10 characters, got {len(format_bytes(n))}"
```

<details>

<summary>
AssertionError: Output exceeds documented 10-character limit
</summary>
```
Input: 1125899906842624000
Result: '1000.00 PiB'
Length: 11 characters

Boundary cases:
  999 * 2**50 = '999.00 PiB' (10 chars)
  1000 * 2**50 = '1000.00 PiB' (11 chars)
  1023 * 2**50 = '1023.00 PiB' (11 chars)

Checking documented guarantee: 'For all values < 2**60, the output is always <= 10 characters'
Is 1125899906842624000 < 2**60? True
Is output <= 10 characters? False
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/46/repo.py", line 31, in <module>
    assert len(format_bytes(n)) <= 10, f"Expected <= 10 characters, got {len(format_bytes(n))}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected <= 10 characters, got 11
```
</details>

## Why This Is A Bug

The function's docstring at line 1788 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py` explicitly guarantees: "For all values < 2**60, the output is always <= 10 characters." This is a clear, unambiguous API contract that developers may rely upon for fixed-width formatting, table layouts, or display constraints.

The bug occurs because the function uses a fixed format string `.2f` which always produces 2 decimal places. When the value reaches 1000 or higher in any unit (e.g., 1000.00 PiB), the numeric portion becomes 7 characters. Combined with the unit suffix " PiB" (4 characters including the space), the total output becomes 11 characters, violating the documented 10-character guarantee.

The violation specifically occurs for values >= 1000 PiB (1.126e18 bytes), which are still well below the documented upper bound of 2**60 (1.153e18 bytes). This means there's a range of valid inputs where the function fails to meet its documented contract.

## Relevant Context

The `format_bytes` function is located in `dask/utils.py` and is part of Dask's utility functions for human-readable formatting. The function iterates through binary prefixes (PiB, TiB, GiB, MiB, kiB) and selects the appropriate unit when the value is >= 90% of that unit's threshold.

Key implementation details:
- Line 1797: The threshold check uses `n >= k * 0.9` to determine unit selection
- Line 1798: The formatting uses `f"{n / k:.2f} {prefix}B"` which always produces 2 decimal places
- The function handles values from 0 bytes to just under 2**60 bytes (1 exbibyte)

While values >= 1000 PiB are rare in practice (representing over 1 exabyte of data), the documented guarantee uses the word "always" without caveats, making this a legitimate contract violation that should be fixed to maintain API reliability.

Documentation link: https://docs.dask.org/en/stable/generated/dask.utils.format_bytes.html

## Proposed Fix

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
+                return f"{value:.1f} {prefix}B"
+            else:
+                return f"{value:.2f} {prefix}B"
     return f"{n} B"
```