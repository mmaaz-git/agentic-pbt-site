# Bug Report: numpy.char.replace() Silently Truncates String Replacements

**Target**: `numpy.char.replace()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.replace()` silently truncates the output when replacements would make strings longer than the original, causing data corruption without any warning or error.

## Property-Based Test

```python
import numpy.char as char
from hypothesis import given, strategies as st, settings

@given(st.text(min_size=1, max_size=50))
@settings(max_examples=500, deadline=None)
def test_replace_matches_python(s):
    for old, new in [(s, s + 'x'), (s[0], s[0] * 2)]:
        numpy_result = char.replace(s, old, new)
        numpy_str = str(numpy_result.item() if hasattr(numpy_result, 'item') else numpy_result)
        python_result = s.replace(old, new)
        assert numpy_str == python_result, f"numpy.char.replace({repr(s)}, {repr(old)}, {repr(new)}) returned {repr(numpy_str)}, expected {repr(python_result)}"

if __name__ == "__main__":
    test_replace_matches_python()
```

<details>

<summary>
**Failing input**: `s='0'`, `old='0'`, `new='0x'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 14, in <module>
    test_replace_matches_python()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 5, in test_replace_matches_python
    @settings(max_examples=500, deadline=None)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/35/hypo.py", line 11, in test_replace_matches_python
    assert numpy_str == python_result, f"numpy.char.replace({repr(s)}, {repr(old)}, {repr(new)}) returned {repr(numpy_str)}, expected {repr(python_result)}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: numpy.char.replace('0', '0', '0x') returned '0', expected '0x'
Falsifying example: test_replace_matches_python(
    s='0',  # or any other generated value
)
Exit code: 1
```
</details>

## Reproducing the Bug

```python
import numpy.char as char
import numpy as np

print("=== Testing numpy.char.replace() truncation bug ===\n")

test_cases = [
    ('a', 'a', 'aa'),
    ('hello', 'hello', 'hello world'),
    ('test', 'test', 'testing'),
    ('Dr', 'Dr', 'Doctor'),
    ('foo', 'o', 'oo'),
    ('x', 'x', 'xyz'),
]

for haystack, old, new in test_cases:
    # Test with numpy.char.replace
    numpy_result = char.replace(haystack, old, new)
    numpy_str = str(numpy_result.item() if hasattr(numpy_result, 'item') else numpy_result)

    # Test with Python's str.replace
    python_result = haystack.replace(old, new)

    print(f"replace({repr(haystack)}, {repr(old)}, {repr(new)})")
    print(f"  numpy result:  {repr(numpy_str)}")
    print(f"  python result: {repr(python_result)}")
    print(f"  numpy dtype:   {numpy_result.dtype}")

    if numpy_str != python_result:
        print(f"  ❌ MISMATCH - Data truncated!")
    else:
        print(f"  ✓ Match")
    print()

print("\n=== Demonstrating the root cause ===\n")
print("The issue is that numpy.char.replace preserves the original dtype:")
print()

# Show how dtype affects the result
test_str = 'a'
for dtype_size in [1, 2, 5]:
    arr = np.array(test_str, dtype=f'<U{dtype_size}')
    result = char.replace(arr, 'a', 'aa')
    print(f"Input dtype: {arr.dtype}, value: {repr(str(arr))}")
    print(f"Output dtype: {result.dtype}, value: {repr(str(result))}")
    print(f"Expected: 'aa', Got: {repr(str(result))}, Correct: {str(result) == 'aa'}")
    print()
```

<details>

<summary>
Silent data truncation in 5 out of 6 test cases
</summary>
```
=== Testing numpy.char.replace() truncation bug ===

replace('a', 'a', 'aa')
  numpy result:  'a'
  python result: 'aa'
  numpy dtype:   <U1
  ❌ MISMATCH - Data truncated!

replace('hello', 'hello', 'hello world')
  numpy result:  'hello'
  python result: 'hello world'
  numpy dtype:   <U5
  ❌ MISMATCH - Data truncated!

replace('test', 'test', 'testing')
  numpy result:  'test'
  python result: 'testing'
  numpy dtype:   <U4
  ❌ MISMATCH - Data truncated!

replace('Dr', 'Dr', 'Doctor')
  numpy result:  'Do'
  python result: 'Doctor'
  numpy dtype:   <U2
  ❌ MISMATCH - Data truncated!

replace('foo', 'o', 'oo')
  numpy result:  'foooo'
  python result: 'foooo'
  numpy dtype:   <U5
  ✓ Match

replace('x', 'x', 'xyz')
  numpy result:  'x'
  python result: 'xyz'
  numpy dtype:   <U1
  ❌ MISMATCH - Data truncated!


=== Demonstrating the root cause ===

The issue is that numpy.char.replace preserves the original dtype:

Input dtype: <U1, value: 'a'
Output dtype: <U1, value: 'a'
Expected: 'aa', Got: 'a', Correct: False

Input dtype: <U2, value: 'a'
Output dtype: <U2, value: 'aa'
Expected: 'aa', Got: 'aa', Correct: True

Input dtype: <U5, value: 'a'
Output dtype: <U2, value: 'aa'
Expected: 'aa', Got: 'aa', Correct: True
```
</details>

## Why This Is A Bug

This behavior violates NumPy's documented contract and expected behavior in multiple ways:

1. **Documentation contradiction**: The function documentation states "For each element in ``a``, return a copy of the string with occurrences of substring ``old`` replaced by ``new``" and references `str.replace` in the "See Also" section. However, it fails to perform complete replacements when they would expand the string length.

2. **Silent data corruption**: The function truncates results without any warning, error, or indication that data has been lost. This is the most dangerous type of bug as it can propagate through data pipelines undetected.

3. **Inconsistent with Python's str.replace**: Despite the explicit reference to `str.replace`, the behavior differs fundamentally. Python's `str.replace('a', 'a', 'aa')` correctly returns `'aa'`, while `numpy.char.replace('a', 'a', 'aa')` returns `'a'`.

4. **Breaks common use cases**: Real-world applications frequently need replacements that expand strings:
   - Abbreviation expansion: `'Dr'` → `'Doctor'`, `'St'` → `'Street'`
   - Text formatting: Adding prefixes/suffixes, wrapping in quotes
   - Data cleaning: Replacing single characters with escaped versions

5. **Dtype preservation over correctness**: The root cause is that NumPy preserves the input's dtype (e.g., `<U1` for single character) in the output, even when this makes the operation mathematically incorrect. A string of dtype `<U1` can only hold 1 character, so any replacement making it longer gets silently truncated.

## Relevant Context

### NumPy Version and Documentation
- Tested on NumPy 2.3.0
- Documentation: https://numpy.org/doc/stable/reference/generated/numpy.char.replace.html
- The function is also available as `numpy.strings.replace()` in newer versions

### Related NumPy String Handling Issues
NumPy uses fixed-width string dtypes (`<U` for Unicode, `<S` for bytes) where the number indicates maximum characters. This design choice creates issues when operations change string length. While this is a known limitation of NumPy's architecture, the `char.replace()` function should either:
1. Automatically allocate sufficient space for the result
2. Raise an error when truncation would occur
3. Clearly document this limitation in its docstring

### Workaround
Users can work around this by pre-allocating arrays with sufficient space:
```python
# Instead of: char.replace('a', 'a', 'aa')
# Use: char.replace(np.array('a', dtype='<U2'), 'a', 'aa')
```
However, this requires knowing the maximum result length in advance, which defeats the purpose of a general-purpose replace function.

## Proposed Fix

The fix requires modifying NumPy's string replacement logic to properly handle output size. Since the underlying issue is architectural (fixed-size arrays), here's a high-level approach:

1. **Calculate required output size**: Before performing replacements, scan the input to count occurrences of `old` and calculate the maximum possible output length:
   ```
   max_len = input_len + count(old) * (len(new) - len(old))
   ```

2. **Allocate appropriately sized output array**: Create the output array with sufficient dtype size to hold the longest possible result.

3. **Alternative: Raise an error on truncation**: If changing output dtype is not feasible, at minimum raise a warning or error when truncation would occur:
   ```python
   if result_would_be_truncated:
       raise ValueError(f"numpy.char.replace would truncate result from {expected_len} to {dtype_len} characters")
   ```

4. **Update documentation**: Clearly document any limitations in the function's docstring, removing the misleading reference to `str.replace` if behavior differs.

The exact implementation would need to be done in NumPy's C code for the string ufuncs, likely in the `_replace` function implementation.