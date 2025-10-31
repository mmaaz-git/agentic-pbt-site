# Bug Report: numpy.char Case Conversion Functions Silently Truncate Expanding Unicode Mappings

**Target**: `numpy.char.upper`, `numpy.char.lower`, `numpy.char.swapcase`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The numpy.char case conversion functions silently truncate results when Unicode characters expand during case conversion, violating their documented contract to call Python's str methods element-wise.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis property-based test for numpy.char case conversion functions."""

from hypothesis import given, strategies as st, settings, example
import numpy as np
import numpy.char as nc

# Test for upper()
@given(st.text(min_size=1))
@example('ß')  # German sharp S
@example('ﬁ')  # Latin small ligature fi
@settings(max_examples=100)
def test_upper_matches_python(s):
    """Test that numpy.char.upper matches Python's str.upper."""
    arr = np.array([s])
    numpy_result = nc.upper(arr)[0]
    python_result = s.upper()
    assert numpy_result == python_result, (
        f"Mismatch for {repr(s)}: "
        f"numpy={repr(numpy_result)}, python={repr(python_result)}"
    )

# Test for lower()
@given(st.text(min_size=1))
@example('İ')  # Turkish capital I with dot
@settings(max_examples=100)
def test_lower_matches_python(s):
    """Test that numpy.char.lower matches Python's str.lower."""
    arr = np.array([s])
    numpy_result = nc.lower(arr)[0]
    python_result = s.lower()
    assert numpy_result == python_result, (
        f"Mismatch for {repr(s)}: "
        f"numpy={repr(numpy_result)}, python={repr(python_result)}"
    )

# Test for swapcase()
@given(st.text(min_size=1))
@example('ß')  # German sharp S
@example('ﬁ')  # Latin small ligature fi
@settings(max_examples=100)
def test_swapcase_matches_python(s):
    """Test that numpy.char.swapcase matches Python's str.swapcase."""
    arr = np.array([s])
    numpy_result = nc.swapcase(arr)[0]
    python_result = s.swapcase()
    assert numpy_result == python_result, (
        f"Mismatch for {repr(s)}: "
        f"numpy={repr(numpy_result)}, python={repr(python_result)}"
    )

if __name__ == "__main__":
    print("Testing numpy.char.upper()...")
    try:
        test_upper_matches_python()
        print("✓ test_upper_matches_python passed")
    except AssertionError as e:
        print(f"✗ test_upper_matches_python failed: {e}")

    print("\nTesting numpy.char.lower()...")
    try:
        test_lower_matches_python()
        print("✓ test_lower_matches_python passed")
    except AssertionError as e:
        print(f"✗ test_lower_matches_python failed: {e}")

    print("\nTesting numpy.char.swapcase()...")
    try:
        test_swapcase_matches_python()
        print("✓ test_swapcase_matches_python passed")
    except AssertionError as e:
        print(f"✗ test_swapcase_matches_python failed: {e}")
```

<details>

<summary>
**Failing input**: `'ß'` (German sharp S) and others
</summary>
```
Testing numpy.char.upper()...
✗ test_upper_matches_python failed: Mismatch for 'ß': numpy=np.str_('S'), python='SS'

Testing numpy.char.lower()...
✗ test_lower_matches_python failed: Mismatch for 'İ': numpy=np.str_('i'), python='i̇'

Testing numpy.char.swapcase()...
✗ test_swapcase_matches_python failed: Mismatch for 'ß': numpy=np.str_('S'), python='SS'

Falsifying example: test_upper_matches_python(
    s='ß',
)
AssertionError: Mismatch for 'ß': numpy=np.str_('S'), python='SS'

Falsifying example: test_lower_matches_python(
    s='İ',
)
AssertionError: Mismatch for 'İ': numpy=np.str_('i'), python='i̇'

Falsifying example: test_swapcase_matches_python(
    s='ß',
)
AssertionError: Mismatch for 'ß': numpy=np.str_('S'), python='SS'
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of numpy.char Unicode expansion truncation bug."""

import numpy as np
import numpy.char as nc

print("NumPy version:", np.__version__)
print()

# Test upper() with expanding characters
print("Testing numpy.char.upper() with expanding Unicode characters:")
print("-" * 60)

test_cases_upper = [
    ('ß', 'German sharp S'),
    ('ﬁ', 'Latin small ligature fi'),
    ('ﬂ', 'Latin small ligature fl'),
    ('ﬀ', 'Latin small ligature ff'),
    ('ﬃ', 'Latin small ligature ffi'),
    ('ﬄ', 'Latin small ligature ffl'),
    ('ﬅ', 'Latin small ligature st'),
    ('ﬆ', 'Latin small ligature st (variant)'),
]

for char, description in test_cases_upper:
    arr = np.array([char])
    numpy_result = nc.upper(arr)[0]
    python_result = char.upper()
    match = '✓' if numpy_result == python_result else '✗'
    print(f"{char} ({description}):")
    print(f"  NumPy:  {repr(numpy_result)} (dtype: {nc.upper(arr).dtype})")
    print(f"  Python: {repr(python_result)}")
    print(f"  Match: {match}")
    print()

# Test lower() with expanding characters
print("\nTesting numpy.char.lower() with expanding Unicode characters:")
print("-" * 60)

test_cases_lower = [
    ('İ', 'Latin capital letter I with dot above (Turkish)'),
]

for char, description in test_cases_lower:
    arr = np.array([char])
    numpy_result = nc.lower(arr)[0]
    python_result = char.lower()
    match = '✓' if numpy_result == python_result else '✗'
    print(f"{char} ({description}):")
    print(f"  NumPy:  {repr(numpy_result)} (dtype: {nc.lower(arr).dtype})")
    print(f"  Python: {repr(python_result)}")
    print(f"  Match: {match}")
    print()

# Test swapcase() with expanding characters
print("\nTesting numpy.char.swapcase() with expanding Unicode characters:")
print("-" * 60)

test_cases_swapcase = [
    ('ß', 'German sharp S'),
    ('ﬁ', 'Latin small ligature fi'),
]

for char, description in test_cases_swapcase:
    arr = np.array([char])
    numpy_result = nc.swapcase(arr)[0]
    python_result = char.swapcase()
    match = '✓' if numpy_result == python_result else '✗'
    print(f"{char} ({description}):")
    print(f"  NumPy:  {repr(numpy_result)} (dtype: {nc.swapcase(arr).dtype})")
    print(f"  Python: {repr(python_result)}")
    print(f"  Match: {match}")
    print()

print("\nSummary:")
print("-" * 60)
print("The issue: NumPy's fixed-width string arrays truncate results when")
print("Unicode characters expand during case conversion. The dtype remains")
print("<U1 (single character) even when the result should be longer.")
```

<details>

<summary>
All tested Unicode expansions fail with silent truncation
</summary>
```
NumPy version: 2.3.0

Testing numpy.char.upper() with expanding Unicode characters:
------------------------------------------------------------
ß (German sharp S):
  NumPy:  np.str_('S') (dtype: <U1)
  Python: 'SS'
  Match: ✗

ﬁ (Latin small ligature fi):
  NumPy:  np.str_('F') (dtype: <U1)
  Python: 'FI'
  Match: ✗

ﬂ (Latin small ligature fl):
  NumPy:  np.str_('F') (dtype: <U1)
  Python: 'FL'
  Match: ✗

ﬀ (Latin small ligature ff):
  NumPy:  np.str_('F') (dtype: <U1)
  Python: 'FF'
  Match: ✗

ﬃ (Latin small ligature ffi):
  NumPy:  np.str_('F') (dtype: <U1)
  Python: 'FFI'
  Match: ✗

ﬄ (Latin small ligature ffl):
  NumPy:  np.str_('F') (dtype: <U1)
  Python: 'FFL'
  Match: ✗

ﬅ (Latin small ligature st):
  NumPy:  np.str_('S') (dtype: <U1)
  Python: 'ST'
  Match: ✗

ﬆ (Latin small ligature st (variant)):
  NumPy:  np.str_('S') (dtype: <U1)
  Python: 'ST'
  Match: ✗


Testing numpy.char.lower() with expanding Unicode characters:
------------------------------------------------------------
İ (Latin capital letter I with dot above (Turkish)):
  NumPy:  np.str_('i') (dtype: <U1)
  Python: 'i̇'
  Match: ✗


Testing numpy.char.swapcase() with expanding Unicode characters:
------------------------------------------------------------
ß (German sharp S):
  NumPy:  np.str_('S') (dtype: <U1)
  Python: 'SS'
  Match: ✗

ﬁ (Latin small ligature fi):
  NumPy:  np.str_('F') (dtype: <U1)
  Python: 'FI'
  Match: ✗


Summary:
------------------------------------------------------------
The issue: NumPy's fixed-width string arrays truncate results when
Unicode characters expand during case conversion. The dtype remains
<U1 (single character) even when the result should be longer.
```
</details>

## Why This Is A Bug

This violates the documented behavior of these functions in multiple ways:

1. **Explicit Contract Violation**: The documentation for `numpy.char.upper()` states "Calls str.upper element-wise", and similar claims exist for `lower()` and `swapcase()`. This creates an explicit contract that these functions should behave identically to Python's str methods.

2. **Silent Data Corruption**: The functions silently truncate expanded results without warning or error. When 'ß'.upper() produces 'SS' in Python, numpy.char.upper() silently truncates it to 'S', losing data.

3. **Real-World Impact**: This affects:
   - German text processing (ß → SS is a standard uppercase transformation)
   - Turkish text (İ with combining dot)
   - Documents with typographic ligatures (ﬁ, ﬂ, ﬀ, etc.)
   - Any text processing that relies on correct Unicode case mappings

4. **Violation of Unicode Standards**: The Unicode standard defines these case mappings, and Python's str methods correctly implement them. NumPy's deviation breaks Unicode compliance.

5. **No Documentation of Limitation**: The documentation provides no warning about this behavior, leaving users to discover data corruption through testing or production failures.

## Relevant Context

The root cause is NumPy's fixed-width string array architecture. When creating an array with `np.array(['ß'])`, NumPy infers the dtype as `<U1` (single Unicode character). The case conversion functions operate within this constraint, unable to expand the dtype to accommodate longer results.

Key observations:
- The issue occurs in NumPy 2.3.0 (and likely earlier versions)
- The actual implementation calls `_vec_string(a_arr, a_arr.dtype, 'upper')` which preserves the input dtype
- Python 3's str.upper() correctly returns 'SS' for 'ß' per Unicode standards
- The truncation happens at the NumPy array storage level, not in the string operation itself

Related documentation:
- [NumPy char module documentation](https://numpy.org/doc/stable/reference/routines.char.html)
- [Unicode case mapping standards](https://www.unicode.org/reports/tr21/tr21-5.html)

## Proposed Fix

The fix requires modifying the case conversion functions to handle expanding mappings. Here's a high-level approach:

1. Pre-scan the input to determine maximum expansion needed
2. Allocate output array with appropriate dtype width
3. Apply the case conversion with proper result storage

Since this involves changes to NumPy's core string handling, a simpler workaround for users is to pre-allocate arrays with sufficient width:

```python
# Workaround: Use wider dtype from the start
arr = np.array(['ß'], dtype='<U10')  # Allocate space for expansion
result = nc.upper(arr)  # Now returns 'SS' correctly
```

A proper fix would require changes to the `_vec_string` function to dynamically determine output dtype size based on the operation being performed, which is a non-trivial change to NumPy's architecture.