# Bug Report: numpy.char Functions Silently Truncate Strings When Operations Expand Length

**Target**: `numpy.char.upper()`, `numpy.char.swapcase()`, `numpy.char.replace()`, `numpy.char.translate()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

NumPy char functions silently truncate results when operations expand strings beyond their original fixed-width dtype, causing data loss for Unicode case mappings, string replacements, and character translations.

## Property-Based Test

```python
import numpy as np
import numpy.char
from hypothesis import given, strategies as st, example

@given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10))
@example(strings=['ﬀ'])  # LATIN SMALL LIGATURE FF that expands to 'FF'
def test_upper_matches_python(strings):
    for s in strings:
        arr = np.array([s])
        numpy_result = numpy.char.upper(arr)[0]
        python_result = s.upper()
        assert numpy_result == python_result, f"numpy.char.upper('{s}') = '{numpy_result}' != '{python_result}' = str.upper()"

if __name__ == "__main__":
    test_upper_matches_python()
```

<details>

<summary>
**Failing input**: `strings=['ﬀ']`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/57
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_upper_matches_python FAILED                                [100%]

=================================== FAILURES ===================================
__________________________ test_upper_matches_python ___________________________

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10))
>   @example(strings=['ﬀ'])  # LATIN SMALL LIGATURE FF that expands to 'FF'
                   ^^^

hypo.py:6:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py:1613: in _raise_to_user
    raise the_error_hypothesis_found
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

strings = ['ﬀ']

    @given(st.lists(st.text(min_size=1, max_size=50), min_size=1, max_size=10))
    @example(strings=['ﬀ'])  # LATIN SMALL LIGATURE FF that expands to 'FF'
    def test_upper_matches_python(strings):
        for s in strings:
            arr = np.array([s])
            numpy_result = numpy.char.upper(arr)[0]
            python_result = s.upper()
>           assert numpy_result == python_result, f"numpy.char.upper('{s}') = '{numpy_result}' != '{python_result}' = str.upper()"
E           AssertionError: numpy.char.upper('ﬀ') = 'F' != 'FF' = str.upper()
E           assert np.str_('F') == 'FF'
E
E             - FF
E             + F
E           Falsifying explicit example: test_upper_matches_python(
E               strings=['ﬀ'],
E           )

hypo.py:12: AssertionError
=========================== short test summary info ============================
FAILED hypo.py::test_upper_matches_python - AssertionError: numpy.char.upper(...
============================== 1 failed in 0.20s ===============================
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.char

print("NumPy char truncation bugs demonstration")
print("=" * 50)

# Bug 1: upper() truncates ligatures
print("\nBug 1: numpy.char.upper() truncates ligatures")
print("-" * 40)
arr = np.array(['ﬀ'])  # LATIN SMALL LIGATURE FF, U+FB00
numpy_result = numpy.char.upper(arr)[0]
python_result = 'ﬀ'.upper()
print(f"Input: 'ﬀ' (LATIN SMALL LIGATURE FF)")
print(f"Python str.upper(): '{python_result}'")
print(f"numpy.char.upper(): '{numpy_result}'")
print(f"Data loss: {len(python_result) - len(numpy_result)} characters truncated")

# Bug 2: swapcase() truncates expansions
print("\nBug 2: numpy.char.swapcase() truncates expansions")
print("-" * 40)
arr = np.array(['ß'])  # German sharp S
numpy_result = numpy.char.swapcase(arr)[0]
python_result = 'ß'.swapcase()
print(f"Input: 'ß' (German sharp S)")
print(f"Python str.swapcase(): '{python_result}'")
print(f"numpy.char.swapcase(): '{numpy_result}'")
print(f"Data loss: {len(python_result) - len(numpy_result)} characters truncated")

# Bug 3: replace() truncates when replacement expands string
print("\nBug 3: numpy.char.replace() truncates expansions")
print("-" * 40)
arr = np.array(['0'])
numpy_result = numpy.char.replace(arr, '0', '00')[0]
python_result = '0'.replace('0', '00')
print(f"Input: '0', replace '0' with '00'")
print(f"Python str.replace(): '{python_result}'")
print(f"numpy.char.replace(): '{numpy_result}'")
print(f"Data loss: {len(python_result) - len(numpy_result)} characters truncated")

# Bug 4: translate() truncates when translation expands characters
print("\nBug 4: numpy.char.translate() truncates expansions")
print("-" * 40)
translation_table = str.maketrans({'a': 'AA'})
arr = np.array(['a'])
numpy_result = numpy.char.translate(arr, translation_table)[0]
python_result = 'a'.translate(translation_table)
print(f"Input: 'a', translate 'a' to 'AA'")
print(f"Python str.translate(): '{python_result}'")
print(f"numpy.char.translate(): '{numpy_result}'")
print(f"Data loss: {len(python_result) - len(numpy_result)} characters truncated")

# Show dtype limitation
print("\nRoot cause: Fixed-width dtype limitation")
print("-" * 40)
test_array = np.array(['ﬀ'])
print(f"Array dtype before upper(): {test_array.dtype}")
result_array = numpy.char.upper(test_array)
print(f"Array dtype after upper(): {result_array.dtype}")
print("Result is truncated to fit original dtype width")
```

<details>

<summary>
Output showing silent data truncation in all four functions
</summary>
```
NumPy char truncation bugs demonstration
==================================================

Bug 1: numpy.char.upper() truncates ligatures
----------------------------------------
Input: 'ﬀ' (LATIN SMALL LIGATURE FF)
Python str.upper(): 'FF'
numpy.char.upper(): 'F'
Data loss: 1 characters truncated

Bug 2: numpy.char.swapcase() truncates expansions
----------------------------------------
Input: 'ß' (German sharp S)
Python str.swapcase(): 'SS'
numpy.char.swapcase(): 'S'
Data loss: 1 characters truncated

Bug 3: numpy.char.replace() truncates expansions
----------------------------------------
Input: '0', replace '0' with '00'
Python str.replace(): '00'
numpy.char.replace(): '0'
Data loss: 1 characters truncated

Bug 4: numpy.char.translate() truncates expansions
----------------------------------------
Input: 'a', translate 'a' to 'AA'
Python str.translate(): 'AA'
numpy.char.translate(): 'A'
Data loss: 1 characters truncated

Root cause: Fixed-width dtype limitation
----------------------------------------
Array dtype before upper(): <U1
Array dtype after upper(): <U1
Result is truncated to fit original dtype width
```
</details>

## Why This Is A Bug

This violates the documented API contract. Each affected function explicitly states it "Calls :meth:`str.X` element-wise":

- **numpy.char.upper()**: "Calls :meth:`str.upper` element-wise" - but 'ﬀ'.upper() returns 'FF' in Python, not 'F'
- **numpy.char.swapcase()**: "Calls :meth:`str.swapcase` element-wise" - but 'ß'.swapcase() returns 'SS' in Python, not 'S'
- **numpy.char.replace()**: Claims to "return a copy of the string with all occurrences of substring old replaced by new" - but fails when replacement expands string length
- **numpy.char.translate()**: "Calls :meth:`str.translate` element-wise" - but truncates translations that expand characters

The functions do not produce the same results as Python's string methods, contradicting their documentation. This causes:

1. **Silent data loss**: Results are truncated without any warning or error
2. **Unicode incorrectness**: Standard Unicode case mappings (ﬀ→FF, ß→SS) are not handled properly
3. **Principle of least surprise violation**: Users reasonably expect numpy.char functions to behave like Python's string methods based on the documentation
4. **Data integrity issues**: Applications relying on these functions for text processing will silently lose data

The root cause is NumPy's fixed-width string arrays (dtype '<U1' for single characters), which cannot accommodate expanded results. However, the documentation makes no mention of this limitation in the function docstrings.

## Relevant Context

NumPy uses fixed-width string dtypes like '<U1' (1-character Unicode string), '<U5' (5-character Unicode string), etc. When a string operation would expand beyond the allocated width, the result is silently truncated to fit.

While NumPy's general dtype documentation mentions truncation behavior, the specific function docstrings make explicit claims about calling Python's string methods "element-wise" without warning about potential differences in behavior.

Common affected Unicode characters:
- Ligatures: ﬀ (U+FB00), ﬁ (U+FB01), ﬂ (U+FB02), ﬃ (U+FB03), ﬄ (U+FB04), ﬅ (U+FB05), ﬆ (U+FB06)
- German sharp S: ß (U+00DF) → SS in uppercase
- Greek letters with special uppercase forms
- Various other Unicode characters with expansion mappings

NumPy version tested: 2.3.0

## Proposed Fix

The fix requires calculating the maximum output width before performing string operations. Here's a high-level approach that would need to be implemented in NumPy's C code:

```diff
# Pseudocode showing the conceptual fix for numpy.char.upper()
def upper(a):
-   # Current implementation: uses input dtype
-   out = np.empty(a.shape, dtype=a.dtype)
-   for i, element in enumerate(a.flat):
-       out.flat[i] = element.upper()  # Truncated to fit dtype
+   # Fixed implementation: calculates required width first
+   max_expanded_len = 0
+   for element in a.flat:
+       expanded = str(element).upper()
+       max_expanded_len = max(max_expanded_len, len(expanded))
+
+   # Create output array with sufficient width
+   out_dtype = f'<U{max_expanded_len}' if a.dtype.kind == 'U' else a.dtype
+   out = np.empty(a.shape, dtype=out_dtype)
+
+   # Now apply upper without truncation
+   for i, element in enumerate(a.flat):
+       out.flat[i] = str(element).upper()
    return out
```

Similar fixes would be needed for `swapcase()`, `replace()`, and `translate()`. The actual implementation would need to be in NumPy's C code in `_core/strings.py` or the underlying C implementations.

Alternative minimal fix: Update documentation to clearly warn about truncation behavior when operations expand string length, though this would be less desirable than fixing the actual behavior.