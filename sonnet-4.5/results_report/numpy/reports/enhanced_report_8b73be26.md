# Bug Report: numpy.char Case Functions Truncate Expanding Unicode Characters

**Target**: `numpy.char.upper`, `numpy.char.capitalize`, `numpy.char.swapcase`, `numpy.char.title`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Multiple numpy.char case conversion functions claim to call Python str methods element-wise but incorrectly truncate Unicode characters that expand to multiple characters during case conversion, causing data loss.

## Property-Based Test

```python
import numpy.char as char
import numpy as np
from hypothesis import given, strategies as st, settings


st_text = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',), blacklist_characters='\x00'),
    min_size=0,
    max_size=20
)


@given(st.lists(st_text, min_size=1, max_size=10).map(lambda lst: np.array(lst, dtype='U')))
@settings(max_examples=1000)
def test_upper_matches_python(arr):
    result = char.upper(arr)
    for i in range(len(arr)):
        assert result[i] == arr[i].upper(), f"Failed on {arr[i]!r}: numpy={result[i]!r}, python={arr[i].upper()!r}"


@given(st.lists(st_text, min_size=1, max_size=10).map(lambda lst: np.array(lst, dtype='U')))
@settings(max_examples=1000)
def test_capitalize_matches_python(arr):
    result = char.capitalize(arr)
    for i in range(len(arr)):
        assert result[i] == arr[i].capitalize(), f"Failed on {arr[i]!r}: numpy={result[i]!r}, python={arr[i].capitalize()!r}"


@given(st.lists(st_text, min_size=1, max_size=10).map(lambda lst: np.array(lst, dtype='U')))
@settings(max_examples=1000)
def test_title_matches_python(arr):
    result = char.title(arr)
    for i in range(len(arr)):
        assert result[i] == arr[i].title(), f"Failed on {arr[i]!r}: numpy={result[i]!r}, python={arr[i].title()!r}"


@given(st.lists(st_text, min_size=1, max_size=10).map(lambda lst: np.array(lst, dtype='U')))
@settings(max_examples=1000)
def test_swapcase_matches_python(arr):
    result = char.swapcase(arr)
    for i in range(len(arr)):
        assert result[i] == arr[i].swapcase(), f"Failed on {arr[i]!r}: numpy={result[i]!r}, python={arr[i].swapcase()!r}"


if __name__ == "__main__":
    print("Testing numpy.char.upper...")
    try:
        test_upper_matches_python()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")

    print("\nTesting numpy.char.capitalize...")
    try:
        test_capitalize_matches_python()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")

    print("\nTesting numpy.char.title...")
    try:
        test_title_matches_python()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")

    print("\nTesting numpy.char.swapcase...")
    try:
        test_swapcase_matches_python()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
```

<details>

<summary>
**Failing input**: `array(['ß'])`
</summary>
```
Testing numpy.char.upper...
  FAILED: Failed on np.str_('ß'): numpy=np.str_('S'), python='SS'

Testing numpy.char.capitalize...
  FAILED: Failed on np.str_('ß'): numpy=np.str_('S'), python='Ss'

Testing numpy.char.title...
  FAILED: Failed on np.str_('ß'): numpy=np.str_('S'), python='Ss'

Testing numpy.char.swapcase...
  FAILED: Failed on np.str_('ß'): numpy=np.str_('S'), python='SS'
```
</details>

## Reproducing the Bug

```python
import numpy.char as char
import numpy as np

# Test multiple characters that expand during case conversion
test_cases = ['ß', 'ﬁ', 'ﬂ', 'ﬆ']

for test_char in test_cases:
    arr = np.array([test_char])

    print(f'\nTesting: {test_char!r} (Unicode U+{ord(test_char):04X})')
    print(f'  upper():      numpy={char.upper(arr)[0]!r:8} Python={test_char.upper()!r:8}  Match: {char.upper(arr)[0] == test_char.upper()}')
    print(f'  capitalize(): numpy={char.capitalize(arr)[0]!r:8} Python={test_char.capitalize()!r:8}  Match: {char.capitalize(arr)[0] == test_char.capitalize()}')
    print(f'  title():      numpy={char.title(arr)[0]!r:8} Python={test_char.title()!r:8}  Match: {char.title(arr)[0] == test_char.title()}')
    print(f'  swapcase():   numpy={char.swapcase(arr)[0]!r:8} Python={test_char.swapcase()!r:8}  Match: {char.swapcase(arr)[0] == test_char.swapcase()}')
```

<details>

<summary>
Output showing data truncation across all affected functions
</summary>
```

Testing: 'ß' (Unicode U+00DF)
  upper():      numpy=np.str_('S') Python='SS'      Match: False
  capitalize(): numpy=np.str_('S') Python='Ss'      Match: False
  title():      numpy=np.str_('S') Python='Ss'      Match: False
  swapcase():   numpy=np.str_('S') Python='SS'      Match: False

Testing: 'ﬁ' (Unicode U+FB01)
  upper():      numpy=np.str_('F') Python='FI'      Match: False
  capitalize(): numpy=np.str_('F') Python='Fi'      Match: False
  title():      numpy=np.str_('F') Python='Fi'      Match: False
  swapcase():   numpy=np.str_('F') Python='FI'      Match: False

Testing: 'ﬂ' (Unicode U+FB02)
  upper():      numpy=np.str_('F') Python='FL'      Match: False
  capitalize(): numpy=np.str_('F') Python='Fl'      Match: False
  title():      numpy=np.str_('F') Python='Fl'      Match: False
  swapcase():   numpy=np.str_('F') Python='FL'      Match: False

Testing: 'ﬆ' (Unicode U+FB06)
  upper():      numpy=np.str_('S') Python='ST'      Match: False
  capitalize(): numpy=np.str_('S') Python='St'      Match: False
  title():      numpy=np.str_('S') Python='St'      Match: False
  swapcase():   numpy=np.str_('S') Python='ST'      Match: False
```
</details>

## Why This Is A Bug

This is a clear contract violation between documented and actual behavior. All four affected functions explicitly document that they "Call :meth:`str.method` element-wise" without any caveats about Unicode limitations:

1. **numpy.char.upper**: Documentation states "Calls :meth:`str.upper` element-wise" (line 1616 of /numpy/_core/strings.py)
2. **numpy.char.capitalize**: Documentation states "Calls :meth:`str.capitalize` element-wise" (line 1737)
3. **numpy.char.title**: Documentation states "Calls :meth:`str.title` element-wise" (line 1768)
4. **numpy.char.swapcase**: Documentation states "Calls :meth:`str.swapcase` element-wise" (line 1678)

The actual behavior demonstrates **data loss** where:
- 'ß' (German eszett, U+00DF) should uppercase to 'SS' but numpy truncates to 'S'
- 'ﬁ' (Latin small ligature fi, U+FB01) should uppercase to 'FI' but numpy truncates to 'F'
- Multiple other Unicode ligatures and special characters are similarly affected

Python's str methods correctly follow the Unicode Standard for case mapping, which defines that certain characters expand when converted. For example, the Unicode Standard explicitly specifies ß → SS for uppercase conversion. Numpy's implementation violates this standard and its own documentation by truncating the result.

The root cause is that all four functions delegate to `_vec_string(a_arr, a_arr.dtype, 'method_name')` from `numpy._core.multiarray`, which appears to be a C implementation that performs character-by-character mapping with a fixed-size output buffer, preventing proper expansion.

## Relevant Context

The Unicode case mapping rules that numpy violates are defined in the Unicode Standard, specifically:
- SpecialCasing.txt defines ß → SS uppercase mapping
- CaseFolding.txt defines ligature expansions

This bug affects real-world text processing for:
- German language text (ß is common in German)
- Historical texts using ligatures
- Proper typography in various languages
- Any internationalized application using numpy for text processing

The implementation is in compiled C code (`_vec_string` in numpy._core.multiarray), making it non-trivial to fix. The dtype system in numpy assumes fixed-width strings, which conflicts with operations that change string length.

Documentation references:
- numpy.char.upper: https://numpy.org/doc/stable/reference/generated/numpy.char.upper.html
- Python str.upper: https://docs.python.org/3/library/stdtypes.html#str.upper
- Unicode Standard Case Mappings: https://www.unicode.org/versions/Unicode15.0.0/ch03.pdf#G33992

## Proposed Fix

Since the issue stems from numpy's fixed-width dtype system conflicting with variable-length results, a proper fix requires significant architectural changes. Here's a high-level approach:

1. **Option A: Use Python's str methods with dynamic sizing** (Recommended)
   - Detect when case conversion will change string length
   - Allocate appropriately sized output array
   - Use actual Python str methods for conversion

2. **Option B: Document the limitation** (If Option A is infeasible)
   - Update documentation to explicitly state Unicode expansion limitations
   - Add warnings when truncation occurs
   - Provide examples of affected characters

3. **Option C: Return object arrays when needed** (Compatibility compromise)
   - Detect expanding characters
   - Return object dtype arrays for such cases
   - Maintain backward compatibility for non-expanding cases

A minimal implementation for Option A would involve replacing the `_vec_string` calls with vectorized Python str method calls that handle dynamic sizing properly. This would ensure full Unicode compliance and match the documented behavior.