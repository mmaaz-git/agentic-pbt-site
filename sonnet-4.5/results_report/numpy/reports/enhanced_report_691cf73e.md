# Bug Report: numpy.char Case Transformation Functions Silently Truncate Unicode Characters

**Target**: `numpy.char.upper()`, `numpy.char.lower()`, `numpy.char.title()`, `numpy.char.capitalize()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NumPy's char module case transformation functions silently truncate results when Unicode case mappings produce strings longer than the input array's dtype capacity, causing data corruption without warning for characters like ß→SS, İ→i̇, and ligatures ﬁ→FI.

## Property-Based Test

```python
import numpy as np
import numpy.char as nc
from hypothesis import given, settings, strategies as st

@given(st.lists(st.text(), min_size=1))
@settings(max_examples=1000)
def test_lower_upper_roundtrip(strings):
    arr = np.array(strings, dtype=str)
    lowered = nc.lower(arr)
    result = nc.lower(nc.upper(arr))
    assert np.array_equal(lowered, result)

# Run the test
if __name__ == "__main__":
    test_lower_upper_roundtrip()
```

<details>

<summary>
**Failing input**: `['ß']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 15, in <module>
    test_lower_upper_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 6, in test_lower_upper_roundtrip
    @settings(max_examples=1000)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/63/hypo.py", line 11, in test_lower_upper_roundtrip
    assert np.array_equal(lowered, result)
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_lower_upper_roundtrip(
    strings=['ß'],
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.char as nc

print("=== DEMONSTRATING NUMPY CHAR CASE TRUNCATION BUG ===")
print()

# Test 1: German eszett (ß)
print("1. German eszett (ß) -> SS truncation:")
print("-" * 40)
arr = np.array(['ß'])
print(f"Input array: {arr}")
print(f"Input dtype: {arr.dtype}")
print(f"Expected (Python): 'ß'.upper() = '{('ß'.upper())}'")
result = nc.upper(arr)
print(f"Actual (NumPy): nc.upper(['ß']) = {result}")
print(f"Result string: '{result[0]}'")
print(f"Length lost: Expected 2 chars, got {len(result[0])} char")
print()

# Test 2: Turkish İ
print("2. Turkish İ (I with dot) -> i̇ truncation:")
print("-" * 40)
arr = np.array(['İ'])
print(f"Input array: {arr}")
print(f"Input dtype: {arr.dtype}")
expected = 'İ'.lower()
print(f"Expected (Python): 'İ'.lower() = '{expected}' (length={len(expected)})")
result = nc.lower(arr)
print(f"Actual (NumPy): nc.lower(['İ']) = {result}")
print(f"Result string: '{result[0]}' (length={len(result[0])})")
print(f"Data lost: combining dot above character")
print()

# Test 3: Ligatures
print("3. Ligature ﬁ -> FI truncation:")
print("-" * 40)
arr = np.array(['ﬁ'])
print(f"Input array: {arr}")
print(f"Input dtype: {arr.dtype}")
print(f"Expected (Python): 'ﬁ'.upper() = '{('ﬁ'.upper())}'")
result = nc.upper(arr)
print(f"Actual (NumPy): nc.upper(['ﬁ']) = {result}")
print(f"Result string: '{result[0]}'")
print(f"Data lost: Second character 'I' completely lost")
print()

# Test 4: Round-trip failure
print("4. Round-trip test failure:")
print("-" * 40)
arr = np.array(['ß'])
print(f"Original: {arr}")
upper = nc.upper(arr)
print(f"After upper: {upper}")
lower = nc.lower(upper)
print(f"After lower(upper): {lower}")
print(f"Expected round-trip: nc.lower(nc.upper(['ß'])) should be ['ß'] or at least ['ss']")
print(f"Actual result: {lower}")
print(f"Data corruption: Original 'ß' became '{lower[0]}'")
print()

# Test 5: Workaround with adequate dtype
print("5. Workaround with adequate dtype:")
print("-" * 40)
arr = np.array(['ß'], dtype='<U10')
print(f"Input array with dtype <U10: {arr}")
print(f"Input dtype: {arr.dtype}")
result = nc.upper(arr)
print(f"nc.upper(['ß']) with <U10: {result}")
print(f"Result string: '{result[0]}' - CORRECT!")
print()

# Test 6: Multiple problematic characters
print("6. Multiple problematic characters in one array:")
print("-" * 40)
arr = np.array(['ß', 'İ', 'ﬁ', 'ﬂ', 'ﬆ'])
print(f"Input array: {arr}")
print(f"Input dtype: {arr.dtype}")
upper = nc.upper(arr)
print(f"nc.upper result: {upper}")
print("Expected vs Actual:")
for i, char in enumerate(['ß', 'İ', 'ﬁ', 'ﬂ', 'ﬆ']):
    print(f"  '{char}'.upper() = '{char.upper()}' -> NumPy: '{upper[i]}'")
print()

print("=== SUMMARY ===")
print("Silent data truncation occurs when Unicode case mappings")
print("produce strings longer than the input array's dtype.")
print("No warning or error is raised, leading to data corruption.")
```

<details>

<summary>
Output demonstrating silent data truncation
</summary>
```
=== DEMONSTRATING NUMPY CHAR CASE TRUNCATION BUG ===

1. German eszett (ß) -> SS truncation:
----------------------------------------
Input array: ['ß']
Input dtype: <U1
Expected (Python): 'ß'.upper() = 'SS'
Actual (NumPy): nc.upper(['ß']) = ['S']
Result string: 'S'
Length lost: Expected 2 chars, got 1 char

2. Turkish İ (I with dot) -> i̇ truncation:
----------------------------------------
Input array: ['İ']
Input dtype: <U1
Expected (Python): 'İ'.lower() = 'i̇' (length=2)
Actual (NumPy): nc.lower(['İ']) = ['i']
Result string: 'i' (length=1)
Data lost: combining dot above character

3. Ligature ﬁ -> FI truncation:
----------------------------------------
Input array: ['ﬁ']
Input dtype: <U1
Expected (Python): 'ﬁ'.upper() = 'FI'
Actual (NumPy): nc.upper(['ﬁ']) = ['F']
Result string: 'F'
Data lost: Second character 'I' completely lost

4. Round-trip test failure:
----------------------------------------
Original: ['ß']
After upper: ['S']
After lower(upper): ['s']
Expected round-trip: nc.lower(nc.upper(['ß'])) should be ['ß'] or at least ['ss']
Actual result: ['s']
Data corruption: Original 'ß' became 's'

5. Workaround with adequate dtype:
----------------------------------------
Input array with dtype <U10: ['ß']
Input dtype: <U10
nc.upper(['ß']) with <U10: ['SS']
Result string: 'SS' - CORRECT!

6. Multiple problematic characters in one array:
----------------------------------------
Input array: ['ß' 'İ' 'ﬁ' 'ﬂ' 'ﬆ']
Input dtype: <U1
nc.upper result: ['S' 'İ' 'F' 'F' 'S']
Expected vs Actual:
  'ß'.upper() = 'SS' -> NumPy: 'S'
  'İ'.upper() = 'İ' -> NumPy: 'İ'
  'ﬁ'.upper() = 'FI' -> NumPy: 'F'
  'ﬂ'.upper() = 'FL' -> NumPy: 'F'
  'ﬆ'.upper() = 'ST' -> NumPy: 'S'

=== SUMMARY ===
Silent data truncation occurs when Unicode case mappings
produce strings longer than the input array's dtype.
No warning or error is raised, leading to data corruption.
```
</details>

## Why This Is A Bug

This violates expected behavior in multiple critical ways:

1. **Documentation contradiction**: The NumPy documentation explicitly states that `upper()` and `lower()` "call str.upper()/str.lower() element-wise". However, when `'ß'.upper()` returns `'SS'` in Python, NumPy silently truncates it to `'S'` when the dtype is `<U1`. This directly contradicts the documented behavior.

2. **Silent data corruption**: The functions silently truncate data without any warning or error. When processing text that contains Unicode characters with expanding case mappings (German ß→SS, Turkish İ→i̇, ligatures ﬁ→FI), users lose data without knowing it occurred.

3. **Mathematical property violation**: The round-trip property `lower(upper(x)) == lower(x)` is broken. For example, `nc.lower(nc.upper(['ß']))` returns `['s']` instead of `['ß']`, fundamentally changing the character.

4. **Default behavior causes corruption**: When users create arrays with `np.array(['ß'])`, NumPy automatically infers dtype `<U1` (one Unicode character). This default behavior directly leads to data corruption when case transformation is applied.

5. **Real-world impact**: This affects major languages:
   - German (90M+ speakers): ß is common in words like "Straße" (street)
   - Turkish (75M+ speakers): İ/i distinction is linguistically significant
   - Typography: Ligatures appear in professional typesetting and historical texts

6. **No escape hatch**: Unlike other NumPy operations that might raise warnings for overflow or precision loss, these functions provide no indication that data is being lost.

## Relevant Context

The Unicode Standard defines case mappings that can change string length:
- **Simple case mappings**: One-to-one character mappings (most common)
- **Full case mappings**: Can produce different length strings (ß→SS, ﬁ→FI)

Python's `str.upper()` and `str.lower()` implement full case mappings per the Unicode standard. NumPy's documentation claims to call these functions but actually implements a truncated version due to dtype constraints.

The issue stems from NumPy's string dtype system where `<U1` means "Unicode string of length 1". When `np.array(['ß'])` is created, NumPy sees one character and creates dtype `<U1`, but doesn't account for case transformations that might need more space.

Relevant NumPy source locations:
- Implementation: `/numpy/_core/strings.py`
- Character array wrapper: `/numpy/_core/defchararray.py`

## Proposed Fix

Since this requires changes to NumPy's C implementation for proper handling, here's a high-level approach:

1. **Immediate mitigation** (Python level):
   - Add a pre-transformation check that detects when output might exceed dtype capacity
   - Raise a warning when truncation will occur
   - Update documentation to prominently warn about this limitation

2. **Proper fix** (requires C-level changes):
   - Pre-scan all input strings to determine maximum output length after case transformation
   - Allocate output array with appropriate dtype (e.g., if input is `<U1` but needs 2 chars, use `<U2`)
   - Alternatively, implement dynamic reallocation when needed

3. **Documentation update** (immediate):
   Add to docstrings of `upper()`, `lower()`, `title()`, `capitalize()`:
   ```
   Warning: For Unicode strings, case transformations may produce results
   longer than the input (e.g., 'ß'.upper() = 'SS'). If the input array's
   dtype cannot accommodate the result, data will be silently truncated.
   Consider using dtype='<U{n}' with sufficient size for your use case.
   ```

The core issue is architectural: NumPy's fixed-size string dtypes conflict with Unicode's variable-length case mappings. A complete fix requires either dynamic dtype adjustment or at minimum, clear warnings when data loss occurs.