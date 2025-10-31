# Bug Report: numpy.base_repr Off-by-One Padding Error for Zero

**Target**: `numpy.base_repr`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `numpy.base_repr` function fails to add the correct number of padding zeros when `number=0`. For all padding values, it adds exactly one fewer zero than requested, violating the documented behavior that padding specifies "Number of zeros padded on the left."

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test for numpy.base_repr padding behavior.
This test verifies that padding adds exactly the specified number of zeros.
"""
import numpy as np
from hypothesis import given, strategies as st, settings, example

@given(st.integers(min_value=0, max_value=10000),
       st.integers(min_value=2, max_value=36),
       st.integers(min_value=1, max_value=20))
@example(number=0, base=2, padding=1)  # The specific failing case
@settings(max_examples=100)
def test_base_repr_padding_adds_exact_zeros(number, base, padding):
    """Test that padding adds exactly N zeros to the left of the representation."""
    repr_with_padding = np.base_repr(number, base=base, padding=padding)
    repr_without_padding = np.base_repr(number, base=base, padding=0)
    expected_length = len(repr_without_padding) + padding
    assert len(repr_with_padding) == expected_length, \
        f"For number={number}, base={base}, padding={padding}: " \
        f"got length {len(repr_with_padding)}, expected {expected_length}. " \
        f"repr_with_padding='{repr_with_padding}', repr_without_padding='{repr_without_padding}'"

if __name__ == "__main__":
    # Run the test
    print("Running property-based test for numpy.base_repr padding...")
    print("=" * 60)
    try:
        test_base_repr_padding_adds_exact_zeros()
        print("All tests passed!")
    except AssertionError as e:
        print(f"Test failed: {e}")
        print("\nThis test found that numpy.base_repr does not consistently")
        print("add the specified number of padding zeros for all inputs.")
```

<details>

<summary>
**Failing input**: `number=0, base=2, padding=1`
</summary>
```
Running property-based test for numpy.base_repr padding...
============================================================
Test failed: For number=0, base=2, padding=1: got length 1, expected 2. repr_with_padding='0', repr_without_padding='0'

This test found that numpy.base_repr does not consistently
add the specified number of padding zeros for all inputs.
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of numpy.base_repr padding bug with zero.
"""
import numpy as np

print("Testing numpy.base_repr padding behavior with number=0")
print("=" * 60)

# Test case 1: The specific failing case from the bug report
print("\nTest 1: number=0, padding=1")
result = np.base_repr(0, padding=1)
print(f"  np.base_repr(0, padding=1) = '{result}'")
print(f"  Expected: '00' (the digit '0' with 1 zero padded on the left)")
print(f"  Got:      '{result}'")
print(f"  Length: {len(result)} (expected: 2)")

# Test case 2: Compare with non-zero number
print("\nTest 2: Comparison with number=1, padding=1")
result_1 = np.base_repr(1, padding=1)
print(f"  np.base_repr(1, padding=1) = '{result_1}'")
print(f"  This correctly adds 1 zero to the left of '1'")

# Test case 3: Multiple padding values with zero
print("\nTest 3: Various padding values with number=0")
for padding in range(0, 5):
    result = np.base_repr(0, padding=padding)
    expected_length = 1 + padding  # '0' plus padding zeros
    print(f"  padding={padding}: '{result}' (length={len(result)}, expected={expected_length})")

# Test case 4: Show the inconsistency clearly
print("\nTest 4: Demonstrating the inconsistency")
print("  For any number N and padding P, we expect:")
print("  len(base_repr(N, padding=P)) = len(base_repr(N, padding=0)) + P")
print()

for num in [0, 1, 5, 10]:
    base_repr_no_pad = np.base_repr(num, padding=0)
    base_repr_with_pad = np.base_repr(num, padding=1)
    expected_len = len(base_repr_no_pad) + 1
    actual_len = len(base_repr_with_pad)
    status = "✓" if expected_len == actual_len else "✗"
    print(f"  num={num:2d}: without padding='{base_repr_no_pad}' (len={len(base_repr_no_pad)})")
    print(f"         with padding=1 ='{base_repr_with_pad}' (len={actual_len}, expected={expected_len}) {status}")

# The assertion that fails
print("\n" + "=" * 60)
print("Assertion test:")
try:
    result = np.base_repr(0, padding=1)
    assert result == '00', f"Expected '00', got '{result}'"
    print("PASS: np.base_repr(0, padding=1) == '00'")
except AssertionError as e:
    print(f"FAIL: {e}")
```

<details>

<summary>
AssertionError: Expected '00', got '0'
</summary>
```
Testing numpy.base_repr padding behavior with number=0
============================================================

Test 1: number=0, padding=1
  np.base_repr(0, padding=1) = '0'
  Expected: '00' (the digit '0' with 1 zero padded on the left)
  Got:      '0'
  Length: 1 (expected: 2)

Test 2: Comparison with number=1, padding=1
  np.base_repr(1, padding=1) = '01'
  This correctly adds 1 zero to the left of '1'

Test 3: Various padding values with number=0
  padding=0: '0' (length=1, expected=1)
  padding=1: '0' (length=1, expected=2)
  padding=2: '00' (length=2, expected=3)
  padding=3: '000' (length=3, expected=4)
  padding=4: '0000' (length=4, expected=5)

Test 4: Demonstrating the inconsistency
  For any number N and padding P, we expect:
  len(base_repr(N, padding=P)) = len(base_repr(N, padding=0)) + P

  num= 0: without padding='0' (len=1)
         with padding=1 ='0' (len=1, expected=2) ✗
  num= 1: without padding='1' (len=1)
         with padding=1 ='01' (len=2, expected=2) ✓
  num= 5: without padding='101' (len=3)
         with padding=1 ='0101' (len=4, expected=4) ✓
  num=10: without padding='1010' (len=4)
         with padding=1 ='01010' (len=5, expected=5) ✓

============================================================
Assertion test:
FAIL: Expected '00', got '0'
```
</details>

## Why This Is A Bug

The NumPy documentation explicitly states that the `padding` parameter is the "Number of zeros padded on the left. Default is 0 (no padding)." This creates a clear contract: when `padding=N`, exactly N zeros should be added to the left of the base representation.

The bug manifests as an off-by-one error exclusively when `number=0`:
- For all non-zero numbers, the function correctly adds exactly N zeros when `padding=N`
- For `number=0`, the function adds N-1 zeros instead of N zeros
- This breaks the invariant: `len(base_repr(n, padding=p)) == len(base_repr(n, padding=0)) + p`

The test output shows the pattern clearly:
- `np.base_repr(0, padding=0)` → `'0'` (correct: no padding)
- `np.base_repr(0, padding=1)` → `'0'` (incorrect: should be `'00'`)
- `np.base_repr(0, padding=2)` → `'00'` (incorrect: should be `'000'`)
- `np.base_repr(0, padding=3)` → `'000'` (incorrect: should be `'0000'`)

## Relevant Context

The bug is located in `/numpy/_core/numeric.py` at line 2213. The implementation uses a `while` loop to build the base representation, but when `number=0`, the loop never executes, leaving the result list empty. The code then uses `res or '0'` as a fallback, but this occurs after padding has been added to the empty list, causing the padding to be lost.

Documentation example from numpy shows: `np.base_repr(7, base=5, padding=3)` returns `'00012'`, demonstrating that exactly 3 zeros are added to '12' (7 in base 5). This sets the expectation that padding adds an exact count of zeros, not a minimum width.

The bug only affects the specific case where `number=0` and `padding > 0`. All other combinations work correctly. This edge case inconsistency makes the function behavior unpredictable and violates the principle of least surprise.

## Proposed Fix

```diff
--- a/numpy/_core/numeric.py
+++ b/numpy/_core/numeric.py
@@ -2204,12 +2204,15 @@ def base_repr(number, base=2, padding=0):
     num = abs(int(number))
     res = []
     while num:
         res.append(digits[num % base])
         num //= base
+    # Handle the case when number is 0
+    if not res:
+        res.append('0')
     if padding:
         res.append('0' * padding)
     if number < 0:
         res.append('-')
-    return ''.join(reversed(res or '0'))
+    return ''.join(reversed(res))
```