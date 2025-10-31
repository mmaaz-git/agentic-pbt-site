# Bug Report: pandas.core.methods.describe.format_percentiles - Violates Uniqueness Contract for Extremely Small Values

**Target**: `pandas.core.methods.describe.format_percentiles`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `format_percentiles` function violates its documented uniqueness contract when handling extremely small percentile values (below ~1e-10), causing distinct input values to collapse to identical formatted strings.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
from pandas.core.methods.describe import format_percentiles

@given(st.lists(st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=50))
@example([0.0, 3.6340605919844266e-284])  # Add the specific failing case
@settings(max_examples=100)
def test_format_percentiles_different_inputs_remain_different(percentiles):
    """
    Property: If two percentiles differ, they should have different formatted strings
    """
    unique_percentiles = list(set(percentiles))

    if len(unique_percentiles) <= 1:
        return

    formatted = format_percentiles(percentiles)
    unique_formatted = set(formatted)

    if len(unique_percentiles) > 1:
        assert len(unique_formatted) > 1, (
            f"Different percentiles collapsed to same format: "
            f"input had {len(unique_percentiles)} unique values, "
            f"but output has only {len(unique_formatted)} unique strings: {unique_formatted}"
        )

if __name__ == "__main__":
    # Run the test
    test_format_percentiles_different_inputs_remain_different()
```

<details>

<summary>
**Failing input**: `[0.0, 3.6340605919844266e-284]`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 28, in <module>
    test_format_percentiles_different_inputs_remain_different()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 5, in test_format_percentiles_different_inputs_remain_different
    @example([0.0, 3.6340605919844266e-284])  # Add the specific failing case
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/42/hypo.py", line 20, in test_format_percentiles_different_inputs_remain_different
    assert len(unique_formatted) > 1, (
           ^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Different percentiles collapsed to same format: input had 2 unique values, but output has only 1 unique strings: {'0%'}
Falsifying explicit example: test_format_percentiles_different_inputs_remain_different(
    percentiles=[0.0, 3.6340605919844266e-284],
)
```
</details>

## Reproducing the Bug

```python
from pandas.core.methods.describe import format_percentiles

# Test case 1: Two different values collapse to same format
percentiles = [0.0, 3.6340605919844266e-284]
result = format_percentiles(percentiles)

print("Test 1: Uniqueness violation")
print(f"Input percentiles: {percentiles}")
print(f"Unique input count: {len(set(percentiles))}")
print(f"Output: {result}")
print(f"Unique output count: {len(set(result))}")

# Verify the issue
try:
    assert len(set(percentiles)) == 2
    assert len(set(result)) == 2
except AssertionError as e:
    print(f"AssertionError: Expected 2 unique outputs, but got {len(set(result))}: {set(result)}")

print("\n" + "="*60 + "\n")

# Test case 2: Non-zero value rounds to 0%
percentiles2 = [1.401298464324817e-45]
result2 = format_percentiles(percentiles2)

print("Test 2: Non-zero rounding to 0%")
print(f"Input: {percentiles2}")
print(f"Output: {result2}")

# Verify the issue
try:
    assert result2[0] != '0%', f"Non-zero percentile {percentiles2[0]} rounded to 0%"
except AssertionError as e:
    print(f"AssertionError: {e}")
```

<details>

<summary>
Uniqueness violation and non-zero values formatting as '0%'
</summary>
```
Test 1: Uniqueness violation
Input percentiles: [0.0, 3.6340605919844266e-284]
Unique input count: 2
Output: ['0%', '0%']
Unique output count: 1
AssertionError: Expected 2 unique outputs, but got 1: {'0%'}

============================================================

Test 2: Non-zero rounding to 0%
Input: [1.401298464324817e-45]
Output: ['0%']
AssertionError: Non-zero percentile 1.401298464324817e-45 rounded to 0%
```
</details>

## Why This Is A Bug

The `format_percentiles` function's docstring explicitly states two guarantees that are violated:

**Contract 1 (lines 1562-1563 of format.py):**
> "Rounding precision is chosen so that: (1) if any two elements of ``percentiles`` differ, they remain different after rounding"

This contract is violated when two distinct input values (0.0 and 3.6340605919844266e-284) both format to the same string '0%'. The function promises to maintain uniqueness, but fails to do so for extremely small differences.

**Contract 2 (line 1564 of format.py):**
> "(2) no entry is *rounded* to 0% or 100%."

This contract is also violated. The non-zero value 1.401298464324817e-45 gets rounded to '0%', even though the documentation explicitly states this should not happen unless the input is exactly 0.0.

The root cause lies in the `get_precision` helper function (lines 1609-1616) which calculates the required decimal precision based on the minimum difference between consecutive percentiles. For extremely small differences, this calculation produces very large precision values that exceed Python's floating-point formatting capabilities, causing the formatted values to collapse to the same representation.

## Relevant Context

The issue occurs specifically in the interaction between two functions:

1. **`format_percentiles`** (lines 1546-1606 in `/pandas/io/formats/format.py`): The main function that formats percentile values
2. **`get_precision`** (lines 1609-1616): Helper function that calculates required decimal precision

The problematic code path:
- When percentiles contain extremely small differences, `get_precision` computes: `prec = -np.floor(np.log10(np.min(diff))).astype(int)`
- For a difference of 3.6e-284, this results in a precision of 283
- Python's float formatting cannot handle such extreme precision values reliably
- The formatted strings collapse to identical representations, violating the uniqueness contract

This function is used internally by pandas' `describe()` method to format percentile labels in statistical summaries. While the edge case is extremely rare in practice, it represents a clear violation of the documented behavior.

## Proposed Fix

Cap the maximum precision at a reasonable value that Python's float formatting can handle reliably:

```diff
--- a/pandas/io/formats/format.py
+++ b/pandas/io/formats/format.py
@@ -1612,7 +1612,9 @@ def get_precision(array: np.ndarray | Sequence[float]) -> int:
     diff = np.ediff1d(array, to_begin=to_begin, to_end=to_end)
     diff = abs(diff)
     prec = -np.floor(np.log10(np.min(diff))).astype(int)
-    prec = max(1, prec)
+    # Cap precision at 15 to avoid formatting issues with extremely small differences
+    # Python's float formatting becomes unreliable beyond this precision
+    prec = max(1, min(prec, 15))
     return prec
```