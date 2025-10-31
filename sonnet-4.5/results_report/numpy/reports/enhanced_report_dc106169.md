# Bug Report: numpy.char.replace() Silent String Truncation on chararray

**Target**: `numpy.char.replace()` on chararray
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.replace()` silently truncates results when operating on a `chararray` and the replacement causes the string to expand beyond the original dtype size, producing incorrect results without any warning.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings, example
import numpy.char as char


@settings(max_examples=200)
@given(st.lists(st.text(min_size=1, max_size=5), min_size=1), st.text(min_size=1, max_size=2), st.text(min_size=1, max_size=5))
@example(strings=['0'], old='0', new='00')  # Explicit example that shows the bug
def test_replace_matches_python(strings, old, new):
    arr = char.array(strings)
    np_replaced = char.replace(arr, old, new)
    for i, s in enumerate(strings):
        py_replaced = s.replace(old, new)
        assert np_replaced[i] == py_replaced, f'{s!r}.replace({old!r}, {new!r}): numpy={np_replaced[i]!r}, python={py_replaced!r}'

if __name__ == "__main__":
    test_replace_matches_python()
```

<details>

<summary>
**Failing input**: `strings=['0'], old='0', new='00'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 16, in <module>
    test_replace_matches_python()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 6, in test_replace_matches_python
    @given(st.lists(st.text(min_size=1, max_size=5), min_size=1), st.text(min_size=1, max_size=2), st.text(min_size=1, max_size=5))
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2062, in wrapped_test
    _raise_to_user(errors, state.settings, [], " in explicit examples")
    ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 1613, in _raise_to_user
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/7/hypo.py", line 13, in test_replace_matches_python
    assert np_replaced[i] == py_replaced, f'{s!r}.replace({old!r}, {new!r}): numpy={np_replaced[i]!r}, python={py_replaced!r}'
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: '0'.replace('0', '00'): numpy='0', python='00'
Falsifying explicit example: test_replace_matches_python(
    strings=['0'],
    old='0',
    new='00',
)
```
</details>

## Reproducing the Bug

```python
import numpy.char as char

# Test the main bug case
arr = char.array(['0'])
result = char.replace(arr, '0', '00')

print("=== Main Bug Case ===")
print(f"Input array: char.array(['0'])")
print(f"Operation: char.replace(arr, '0', '00')")
print(f"Expected result: '00' (as Python's str.replace would produce)")
print(f"Actual result: {result[0]!r}")
print(f"Dtype of input: {arr.dtype}")
print(f"Dtype of output: {result.dtype}")

# Test additional examples to show the truncation pattern
print("\n=== Additional Examples ===")
test_cases = [
    ('a', 'a', 'aa'),
    ('ab', 'b', 'bbb'),
    ('x', 'x', 'xyz'),
    ('hello', 'l', 'LL'),
]

for s, old, new in test_cases:
    arr = char.array([s])
    result = char.replace(arr, old, new)
    expected = s.replace(old, new)
    match = result[0] == expected
    print(f"\nInput: {s!r} (dtype: {arr.dtype})")
    print(f"  .replace({old!r}, {new!r})")
    print(f"  Expected: {expected!r}")
    print(f"  Got: {result[0]!r}")
    print(f"  Match: {match}")

# Show that regular numpy arrays with adequate dtype work correctly
print("\n=== Workaround with explicit dtype ===")
import numpy as np
arr = np.array(['0'], dtype='U10')
result = char.replace(arr, '0', '00')
print(f"np.array(['0'], dtype='U10')")
print(f"char.replace(arr, '0', '00')")
print(f"Result: {result[0]!r} (correct)")

# Final assertion that demonstrates the bug
print("\n=== Assertion that fails due to bug ===")
arr = char.array(['0'])
result = char.replace(arr, '0', '00')
try:
    assert result[0] == '00', f"Expected '00' but got {result[0]!r}"
except AssertionError as e:
    print(f"AssertionError: {e}")
```

<details>

<summary>
Output showing truncation behavior
</summary>
```
=== Main Bug Case ===
Input array: char.array(['0'])
Operation: char.replace(arr, '0', '00')
Expected result: '00' (as Python's str.replace would produce)
Actual result: '0'
Dtype of input: <U1
Dtype of output: <U1

=== Additional Examples ===

Input: 'a' (dtype: <U1)
  .replace('a', 'aa')
  Expected: 'aa'
  Got: 'a'
  Match: False

Input: 'ab' (dtype: <U2)
  .replace('b', 'bbb')
  Expected: 'abbb'
  Got: 'abb'
  Match: False

Input: 'x' (dtype: <U1)
  .replace('x', 'xyz')
  Expected: 'xyz'
  Got: 'x'
  Match: False

Input: 'hello' (dtype: <U5)
  .replace('l', 'LL')
  Expected: 'heLLLLo'
  Got: 'heLLLLo'
  Match: True

=== Workaround with explicit dtype ===
np.array(['0'], dtype='U10')
char.replace(arr, '0', '00')
Result: np.str_('00') (correct)

=== Assertion that fails due to bug ===
AssertionError: Expected '00' but got '0'
```
</details>

## Why This Is A Bug

This violates expected behavior for the following reasons:

1. **Silent Data Corruption**: The function silently truncates results without any warning, which can lead to data loss. When char.replace() operates on a chararray, it preserves the original dtype size even when the replacement would expand the string beyond that size.

2. **Documented Contract Violation**: The numpy.char.replace() documentation states it returns "a copy of the string with occurrences of substring old replaced by new" and includes "See Also: str.replace", implying behavioral similarity with Python's str.replace(). The truncation behavior contradicts this expectation.

3. **Inconsistent Behavior**: When using regular numpy arrays with adequate dtype (e.g., dtype='U10'), char.replace() works correctly and produces the expected expanded strings. The bug only manifests with chararray objects that infer minimal dtype sizes.

4. **No Truncation Warning**: The documentation provides no warning about potential truncation when replacement strings exceed the original dtype capacity. Users reasonably expect string replacement to work correctly regardless of string expansion.

5. **Dtype Preservation Issue**: When char.array() creates a chararray, it infers the minimal dtype from input strings (e.g., '<U1' for single-character strings). The char.replace() function then preserves this dtype during replacement operations, causing truncation when len(new) > len(old).

## Relevant Context

The chararray class is explicitly marked as "provided for numarray backward-compatibility" and "not recommended for new development" in the numpy documentation (numpy/_core/defchararray.py:424-430). However, it remains part of the public API and should still function correctly.

The documentation at numpy/_core/defchararray.py:1007-1017 shows that chararray.replace() delegates to the module-level replace function, which is imported from numpy.strings. The truncation occurs because the output array maintains the same dtype as the input chararray.

Documentation reference: https://numpy.org/doc/stable/reference/generated/numpy.char.replace.html

The workaround is straightforward: use regular numpy arrays with an explicit dtype that can accommodate the expanded strings:
```python
# Instead of:
arr = char.array(['text'])

# Use:
arr = np.array(['text'], dtype='U100')  # or appropriate size
```

## Proposed Fix

The fix requires char.replace() to compute an appropriate output dtype when operating on chararrays that may need expansion. Since the exact implementation depends on numpy.strings internals, here's a high-level approach:

1. Before performing the replacement, calculate the maximum possible expansion
2. If the replacement could expand strings beyond the current dtype capacity, create a new output array with sufficient dtype size
3. Perform the replacement operation on the appropriately-sized array
4. Return the result with the correct dtype

This ensures that char.replace() produces correct results even when strings expand, maintaining compatibility with Python's str.replace() semantics while preserving backward compatibility for existing code that doesn't trigger expansion.