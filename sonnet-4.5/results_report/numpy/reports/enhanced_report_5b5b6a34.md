# Bug Report: numpy.char.upper Silently Truncates Unicode Characters That Expand When Uppercased

**Target**: `numpy.char.upper`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.upper` silently truncates Unicode characters that expand when converted to uppercase, violating its documented contract of calling `str.upper` element-wise and causing data corruption without warning.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_upper_lower_unicode(strings):
    arr = np.array(strings, dtype=str)
    upper_result = char.upper(arr)

    for i in range(len(strings)):
        assert upper_result[i] == strings[i].upper()

if __name__ == "__main__":
    test_upper_lower_unicode()
```

<details>

<summary>
**Failing input**: `strings=['ŉ']`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/0
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_upper_lower_unicode FAILED                                 [100%]

=================================== FAILURES ===================================
___________________________ test_upper_lower_unicode ___________________________
hypo.py:6: in test_upper_lower_unicode
    def test_upper_lower_unicode(strings):
                   ^^^
hypo.py:11: in test_upper_lower_unicode
    assert upper_result[i] == strings[i].upper()
E   AssertionError: assert np.str_('ʼ') == 'ʼN'
E
E     - ʼN
E     + ʼ
E   Falsifying example: test_upper_lower_unicode(
E       strings=['ŉ'],
E   )
=========================== short test summary info ============================
FAILED hypo.py::test_upper_lower_unicode - AssertionError: assert np.str_('ʼ'...
============================== 1 failed in 0.26s ===============================
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

# Test case with German eszett that expands when uppercased
arr = np.array(['ß'], dtype=str)
result = char.upper(arr)

print(f"Input: {arr[0]!r} (dtype: {arr.dtype})")
print(f"Result: {result[0]!r} (dtype: {result.dtype})")
print(f"Expected (Python str.upper): {'ß'.upper()!r}")

# Show the mismatch
try:
    assert result[0] == 'SS', f"Expected 'SS', got {result[0]!r}"
except AssertionError as e:
    print(f"\nAssertion failed: {e}")

# Test additional Unicode characters that expand
print("\n--- Additional examples of truncation ---")
test_cases = [
    'ﬁ',  # Latin Small Ligature Fi -> FI
    'ﬂ',  # Latin Small Ligature Fl -> FL
    'ﬃ', # Latin Small Ligature Ffi -> FFI
    'ﬄ', # Latin Small Ligature Ffl -> FFL
    'ﬅ',  # Latin Small Ligature Long S T -> ST
    'ﬆ',  # Latin Small Ligature St -> ST
]

for char_input in test_cases:
    arr = np.array([char_input], dtype=str)
    result = char.upper(arr)
    expected = char_input.upper()
    print(f"Input: {char_input!r} -> numpy: {result[0]!r}, expected: {expected!r}")
```

<details>

<summary>
Silent data truncation demonstrated
</summary>
```
Input: np.str_('ß') (dtype: <U1)
Result: np.str_('S') (dtype: <U1)
Expected (Python str.upper): 'SS'

Assertion failed: Expected 'SS', got np.str_('S')

--- Additional examples of truncation ---
Input: 'ﬁ' -> numpy: np.str_('F'), expected: 'FI'
Input: 'ﬂ' -> numpy: np.str_('F'), expected: 'FL'
Input: 'ﬃ' -> numpy: np.str_('F'), expected: 'FFI'
Input: 'ﬄ' -> numpy: np.str_('F'), expected: 'FFL'
Input: 'ﬅ' -> numpy: np.str_('S'), expected: 'ST'
Input: 'ﬆ' -> numpy: np.str_('S'), expected: 'ST'
```
</details>

## Why This Is A Bug

This violates the documented contract of `numpy.char.upper` which explicitly states it "Calls str.upper element-wise". The function fails to produce the same results as Python's `str.upper()` for legitimate Unicode characters:

1. **Silent Data Corruption**: The function truncates results without any warning or error, leading to incorrect data that may go unnoticed
2. **Unicode Standard Violation**: The Unicode Standard recognizes that case mappings can result in character expansion (e.g., German 'ß' → 'SS', ligatures 'ﬁ' → 'FI')
3. **Inconsistent with Python Behavior**: Python's `str.upper()` correctly handles these expansions ('ß'.upper() == 'SS'), but numpy.char.upper silently truncates to 'S'
4. **Affects Real-World Usage**: German text commonly uses 'ß', and many documents contain ligatures that would be corrupted
5. **No Documentation of Limitation**: The documentation does not warn about this truncation behavior

The root cause is that `numpy.char.upper` maintains the same dtype size as the input array (e.g., '<U1' for single characters) without checking if the uppercase conversion requires more space. When a character expands during uppercase conversion, only the first character of the result is kept.

## Relevant Context

The implementation in `/home/npc/miniconda/lib/python3.13/site-packages/numpy/_core/strings.py:1102-1136` shows that `upper` uses `_vec_string` with the same dtype as the input array, causing the truncation.

Other numpy.char functions like `multiply` and `replace` correctly handle dtype resizing by calculating the required buffer size before operation. The `upper` function should similarly calculate the maximum possible output length after case conversion.

Documentation: https://numpy.org/doc/stable/reference/generated/numpy.char.upper.html

## Proposed Fix

The function should calculate the maximum possible output length for each string after case conversion and allocate an appropriately sized output array, similar to how `multiply` handles it:

```diff
@set_module("numpy.strings")
@array_function_dispatch(_unary_op_dispatcher)
def upper(a):
    """
    Return an array with the elements converted to uppercase.

    Calls :meth:`str.upper` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array-like, with ``StringDType``, ``bytes_``, or ``str_`` dtype
        Input array.

    Returns
    -------
    out : ndarray
        Output array of ``StringDType``, ``bytes_`` or ``str_`` dtype,
        depending on input types

    See Also
    --------
    str.upper

    Examples
    --------
    >>> import numpy as np
    >>> c = np.array(['a1b c', '1bca', 'bca1']); c
    array(['a1b c', '1bca', 'bca1'], dtype='<U5')
    >>> np.strings.upper(c)
    array(['A1B C', '1BCA', 'BCA1'], dtype='<U5')

    """
    a_arr = np.asarray(a)
-   return _vec_string(a_arr, a_arr.dtype, 'upper')
+   # Calculate maximum possible output size after uppercase conversion
+   # This requires checking each string for potential expansion
+   temp_result = _vec_string(a_arr, np.object_, 'upper')
+   max_len = max(len(s) for s in temp_result.flat)
+
+   # Create output array with appropriate size
+   if a_arr.dtype.char == "T":
+       return temp_result.astype(type(a_arr.dtype))
+
+   out_dtype = f"{a_arr.dtype.char}{max_len}"
+   return temp_result.astype(out_dtype)
```