# Bug Report: dask.utils.format_bytes Output Length Contract Violation

**Target**: `dask.utils.format_bytes`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_bytes` function violates its documented contract that "For all values < 2**60, the output is always <= 10 characters" when formatting values >= 1000 PiB (approximately 1.126 exabytes).

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from dask.utils import format_bytes

@given(st.integers(min_value=0, max_value=2**60 - 1))
@settings(max_examples=1000)
def test_format_bytes_output_length_invariant(n):
    result = format_bytes(n)
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"

if __name__ == "__main__":
    test_format_bytes_output_length_invariant()
```

<details>

<summary>
**Failing input**: `1_125_894_277_343_089_729`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 14, in <module>
    test_format_bytes_output_length_invariant()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 8, in test_format_bytes_output_length_invariant
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/6/hypo.py", line 11, in test_format_bytes_output_length_invariant
    assert len(result) <= 10, f"format_bytes({n}) = '{result}' has length {len(result)} > 10"
           ^^^^^^^^^^^^^^^^^
AssertionError: format_bytes(1125894277343089729) = '1000.00 PiB' has length 11 > 10
Falsifying example: test_format_bytes_output_length_invariant(
    n=1_125_894_277_343_089_729,
)
```
</details>

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

<details>

<summary>
Output shows 11-character string violating 10-character limit
</summary>
```
format_bytes(1125894277343089729) = '1000.00 PiB'
Length: 11 characters
Expected: <= 10 characters
Violates documented invariant: True
```
</details>

## Why This Is A Bug

The function's docstring at line 1788 of `/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages/dask/utils.py` explicitly promises: "For all values < 2**60, the output is always <= 10 characters." This is an unambiguous contract that users may rely on for:

- Fixed-width terminal output formatting
- UI layout calculations where column widths matter
- Buffer size allocations in systems expecting predictable output lengths
- Database field width constraints

The violation occurs because the function doesn't handle the edge case where values in PiB units exceed 999.99. When `n / 2**50 >= 1000`, the formatted string becomes "1000.00 PiB" (11 characters) instead of staying within the promised 10-character limit. This affects all values from approximately 1,125,899,906,842,624,000 bytes (1000 PiB) up to 2^60 - 1 bytes (1024 PiB).

The formatting logic at line 1798 uses `f"{n / k:.2f} {prefix}B"` which produces:
- "999.99 PiB" = 10 characters (OK)
- "1000.00 PiB" = 11 characters (violates contract)
- "1024.00 PiB" = 11 characters (the maximum value < 2^60)

## Relevant Context

The `format_bytes` function is used throughout Dask for human-readable byte size displays in logging, monitoring, and diagnostic outputs. The documented 10-character limit appears to be an intentional design constraint, likely for creating aligned tabular output in CLI tools and dashboards.

The function correctly handles all other unit transitions (B → kiB → MiB → GiB → TiB → PiB) by switching units at 0.9 × unit_size, but doesn't account for when the PiB value itself becomes 4 digits.

Relevant source code location: https://github.com/dask/dask/blob/main/dask/utils.py#L1771-L1799

Official documentation: https://docs.dask.org/en/stable/api.html#dask.utils.format_bytes

## Proposed Fix

```diff
def format_bytes(n: int) -> str:
    """Format bytes as text

    [docstring content...]

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
+           # Ensure output stays within 10 characters for values >= 1000
+           if n >= k * 1000:
+               # For values >= 1000 in current unit, format without decimals
+               return f"{n / k:.0f} {prefix}B"
            return f"{n / k:.2f} {prefix}B"
    return f"{n} B"
```