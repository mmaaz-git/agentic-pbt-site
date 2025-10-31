# Bug Report: pandas.io.parsers.read_csv Integer Overflow Silent Data Corruption

**Target**: `pandas.io.parsers.read_csv`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When reading CSV data with `dtype` specified as `int32`, values that exceed the int32 range (-2147483648 to 2147483647) silently overflow/wraparound instead of raising an error, causing data corruption without any warning to the user.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, Verbosity
import pandas as pd
from io import StringIO
import numpy as np

@given(
    st.lists(
        st.tuples(st.integers(), st.integers()),
        min_size=1,
        max_size=50
    )
)
@settings(verbosity=Verbosity.verbose, max_examples=100)
def test_dtype_specification_preserved(rows):
    """Test that dtype specification is correctly preserved without data corruption."""
    csv_string = "a,b\n" + "\n".join(f"{a},{b}" for a, b in rows)
    result = pd.read_csv(StringIO(csv_string), dtype={'a': 'int64', 'b': 'int32'})

    # Check that dtypes are preserved
    assert result['a'].dtype == np.int64
    assert result['b'].dtype == np.int32

    # Check that values are within expected ranges for int32
    for idx, (a_orig, b_orig) in enumerate(rows):
        b_parsed = result['b'].iloc[idx]

        # For int32, values should either:
        # 1. Be correctly parsed if within range
        # 2. Raise an error if out of range
        # They should NOT silently wraparound

        if -2147483648 <= b_orig <= 2147483647:
            # Value is within int32 range, should be preserved
            assert b_parsed == b_orig, f"Value {b_orig} within int32 range but got {b_parsed}"
        else:
            # Value is outside int32 range
            # This SHOULD raise an error, but due to the bug it wraps around
            # We'll check if wraparound occurred
            if b_parsed != b_orig:
                print(f"\n⚠️ BUG DETECTED: Integer overflow!")
                print(f"  Original value: {b_orig}")
                print(f"  Parsed value:   {b_parsed}")
                print(f"  Expected: OverflowError or ValueError")
                print(f"  Actual: Silent wraparound occurred")
                raise AssertionError(
                    f"Integer overflow: {b_orig} silently wrapped to {b_parsed} "
                    f"instead of raising an error"
                )

# Run the test
if __name__ == "__main__":
    print("Running Hypothesis test for pandas.read_csv integer overflow...")
    print("=" * 60)
    try:
        test_dtype_specification_preserved()
        print("\n✓ All tests passed (no overflow detected in random samples)")
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        print("\nThis demonstrates the bug where values outside int32 range")
        print("silently wraparound instead of raising an error.")
```

<details>

<summary>
**Failing input**: `rows=[(0, 2147483648)]`
</summary>
```
Running Hypothesis test for pandas.read_csv integer overflow...
============================================================
Trying example: test_dtype_specification_preserved(
    rows=[(0, 2_147_483_648)],
)

⚠️ BUG DETECTED: Integer overflow!
  Original value: 2147483648
  Parsed value:   -2147483648
  Expected: OverflowError or ValueError
  Actual: Silent wraparound occurred
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 45, in test_dtype_specification_preserved
    raise AssertionError(
    ...<2 lines>...
    )
AssertionError: Integer overflow: 2147483648 silently wrapped to -2147483648 instead of raising an error
```
</details>

## Reproducing the Bug

```python
import pandas as pd
from io import StringIO
import numpy as np

# Test case 1: Basic overflow case
csv = "value\n2147483648"
result = pd.read_csv(StringIO(csv), dtype={'value': 'int32'})

print("=== Basic Overflow Case ===")
print(f"Input value:  2147483648")
print(f"int32 max:    {np.iinfo(np.int32).max} (2^31 - 1)")
print(f"Result value: {result['value'].iloc[0]}")
print(f"Expected: OverflowError or ValueError")
print(f"Actual: Silent wraparound to int32 min\n")

# Test case 2: Multiple overflow values
csv_multi = "id,value\n1,2147483647\n2,2147483648\n3,2147483649\n4,-2147483648\n5,-2147483649"
result_multi = pd.read_csv(StringIO(csv_multi), dtype={'id': 'int32', 'value': 'int32'})

print("=== Multiple Overflow Cases ===")
print("Input CSV:")
print(csv_multi)
print("\nResult DataFrame:")
print(result_multi)
print("\nObservations:")
print(f"  2147483647 (int32_max) -> {result_multi['value'].iloc[0]} ✓")
print(f"  2147483648 (int32_max+1) -> {result_multi['value'].iloc[1]} (wrapped to int32_min)")
print(f"  2147483649 (int32_max+2) -> {result_multi['value'].iloc[2]} (wrapped)")
print(f"  -2147483648 (int32_min) -> {result_multi['value'].iloc[3]} ✓")
print(f"  -2147483649 (int32_min-1) -> {result_multi['value'].iloc[4]} (wrapped to int32_max)\n")

# Test case 3: Very large values (beyond int64)
try:
    csv_huge = "value\n18446744073709551616"  # 2^64
    result_huge = pd.read_csv(StringIO(csv_huge), dtype={'value': 'int32'})
    print("=== Very Large Value Case ===")
    print(f"Input: 18446744073709551616 (2^64)")
    print(f"Result: {result_huge['value'].iloc[0]}")
except OverflowError as e:
    print("=== Very Large Value Case ===")
    print(f"Input: 18446744073709551616 (2^64)")
    print(f"Result: OverflowError raised (correct behavior)")
    print(f"Error message: {e}\n")

# Test case 4: Demonstrate data corruption in realistic scenario
csv_financial = "transaction_id,amount_cents\n1,2000000000\n2,2147483648\n3,2500000000"
result_financial = pd.read_csv(StringIO(csv_financial), dtype={'transaction_id': 'int32', 'amount_cents': 'int32'})

print("=== Financial Data Corruption Example ===")
print("Scenario: Processing transaction amounts in cents")
print("Input data (amounts in cents):")
print("  Transaction 1: $20,000,000.00 (2,000,000,000 cents)")
print("  Transaction 2: $21,474,836.48 (2,147,483,648 cents)")
print("  Transaction 3: $25,000,000.00 (2,500,000,000 cents)")
print("\nResult after parsing with int32:")
for i in range(3):
    original = [2000000000, 2147483648, 2500000000][i]
    parsed = result_financial['amount_cents'].iloc[i]
    print(f"  Transaction {i+1}: {original:,} cents -> {parsed:,} cents")
    if parsed < 0:
        print(f"    ⚠️  CORRUPTED: Negative amount! Lost ${abs(int(original) - int(parsed)) / 100:,.2f}")
```

<details>

<summary>
Silent integer overflow causes data corruption
</summary>
```
=== Basic Overflow Case ===
Input value:  2147483648
int32 max:    2147483647 (2^31 - 1)
Result value: -2147483648
Expected: OverflowError or ValueError
Actual: Silent wraparound to int32 min

=== Multiple Overflow Cases ===
Input CSV:
id,value
1,2147483647
2,2147483648
3,2147483649
4,-2147483648
5,-2147483649

Result DataFrame:
   id       value
0   1  2147483647
1   2 -2147483648
2   3 -2147483647
3   4 -2147483648
4   5  2147483647

Observations:
  2147483647 (int32_max) -> 2147483647 ✓
  2147483648 (int32_max+1) -> -2147483648 (wrapped to int32_min)
  2147483649 (int32_max+2) -> -2147483647 (wrapped)
  -2147483648 (int32_min) -> -2147483648 ✓
  -2147483649 (int32_min-1) -> 2147483647 (wrapped to int32_max)

=== Very Large Value Case ===
Input: 18446744073709551616 (2^64)
Result: OverflowError raised (correct behavior)
Error message: Overflow

=== Financial Data Corruption Example ===
Scenario: Processing transaction amounts in cents
Input data (amounts in cents):
  Transaction 1: $20,000,000.00 (2,000,000,000 cents)
  Transaction 2: $21,474,836.48 (2,147,483,648 cents)
  Transaction 3: $25,000,000.00 (2,500,000,000 cents)

Result after parsing with int32:
  Transaction 1: 2,000,000,000 cents -> 2,000,000,000 cents
  Transaction 2: 2,147,483,648 cents -> -2,147,483,648 cents
    ⚠️  CORRUPTED: Negative amount! Lost $42,949,672.96
  Transaction 3: 2,500,000,000 cents -> -1,794,967,296 cents
    ⚠️  CORRUPTED: Negative amount! Lost $42,949,672.96
```
</details>

## Why This Is A Bug

This is a serious data integrity bug for several reasons:

1. **Silent Data Corruption**: Values silently wraparound without any warning or error. A value of 2,147,483,648 becomes -2,147,483,648 with no indication that overflow occurred.

2. **Inconsistent Behavior**: The bug shows inconsistent overflow handling:
   - Values slightly beyond int32 range (e.g., 2,147,483,648) silently wraparound
   - Values far beyond int64 range (e.g., 18,446,744,073,709,551,616) correctly raise OverflowError
   - This inconsistency violates the principle of least surprise

3. **Violates User Expectations**: When explicitly specifying a dtype, users expect either successful conversion or an error - not silent data corruption.

4. **Real-World Impact**: This affects critical use cases:
   - Financial data: Transaction amounts can become negative
   - Scientific data: Measurements can flip signs
   - ID values: Can become negative or wrap to existing IDs
   - The example shows $42.9M being "lost" due to wraparound

5. **Security Implications**: Integer overflow is a well-known class of security vulnerabilities that can lead to exploitable conditions.

6. **Documentation Gap**: The pandas documentation does not mention or warn about this wraparound behavior, leading users to assume type safety.

## Relevant Context

The bug occurs in the dtype conversion pipeline within pandas:

1. CSV values are initially parsed as strings by the C parser (`pandas._libs.parsers.TextReader`)
2. String values are converted to int64 via `lib.maybe_convert_numeric()`
3. When user specifies `dtype={'col': 'int32'}`, the conversion happens in `_cast_types()`
4. `_cast_types()` calls `astype_array()` which calls `_astype_nansafe()`
5. `_astype_nansafe()` directly calls numpy's `arr.astype(dtype, copy=True)` at line 133 of `/pandas/core/dtypes/astype.py`
6. NumPy's `astype()` silently wraps around on overflow when converting from int64 to int32

The inconsistent behavior arises because:
- Very large values that can't fit in int64 are caught during initial parsing
- Values that fit in int64 but not int32 reach numpy's `astype()` which silently wraps

Related issue: pandas GitHub issue #47167 acknowledges this as a known problem.

## Proposed Fix

The fix should add overflow checking in `_astype_nansafe()` before calling numpy's `astype()`. Here's a patch:

```diff
--- a/pandas/core/dtypes/astype.py
+++ b/pandas/core/dtypes/astype.py
@@ -128,6 +128,21 @@ def _astype_nansafe(
         )
         raise ValueError(msg)

+    # Check for integer overflow when downcasting integer types
+    if arr.dtype.kind in "iu" and dtype.kind in "iu":
+        if dtype.itemsize < arr.dtype.itemsize:
+            # Downcasting - check for overflow
+            info = np.iinfo(dtype)
+            min_val, max_val = info.min, info.max
+
+            # Check if any values are outside the target dtype range
+            if arr.size > 0:
+                arr_min = arr.min()
+                arr_max = arr.max()
+                if arr_min < min_val or arr_max > max_val:
+                    raise OverflowError(
+                        f"Values are outside the range [{min_val}, {max_val}] for dtype {dtype}"
+                    )
+
     if copy or arr.dtype == object or dtype == object:
         # Explicit copy, or required since NumPy can't view from / to object.
         return arr.astype(dtype, copy=True)
```

This patch adds overflow detection before the problematic `astype()` call, ensuring that users get a clear error message instead of silent data corruption.
