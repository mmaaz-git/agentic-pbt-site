# Bug Report: numpy.strings.slice Returns Wrong Result When stop=None

**Target**: `numpy.strings.slice`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.strings.slice()` returns incorrect results when `stop=None` is explicitly passed with a non-None start value, violating Python's standard slice semantics where `None` means "slice to the end".

## Property-Based Test

```python
import numpy as np
import numpy.strings as nps
from hypothesis import assume, given, strategies as st

@given(st.text(min_size=1), st.integers(min_value=0, max_value=20))
def test_slice_with_explicit_stop(s, start):
    assume(start < len(s))
    arr = np.array([s])
    result = nps.slice(arr, start, None)[0]
    expected = s[start:None]
    assert str(result) == expected, f"For s={repr(s)}, start={start}: expected {repr(expected)}, got {repr(str(result))}"

# Run the test
if __name__ == "__main__":
    test_slice_with_explicit_stop()
```

<details>

<summary>
**Failing input**: `s='0', start=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 15, in <module>
    test_slice_with_explicit_stop()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 6, in test_slice_with_explicit_stop
    def test_slice_with_explicit_stop(s, start):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/2/hypo.py", line 11, in test_slice_with_explicit_stop
    assert str(result) == expected, f"For s={repr(s)}, start={start}: expected {repr(expected)}, got {repr(str(result))}"
           ^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: For s='0', start=0: expected '0', got ''
Falsifying example: test_slice_with_explicit_stop(
    s='0',  # or any other generated value
    start=0,
)
```
</details>

## Reproducing the Bug

```python
import numpy as np
import numpy.strings as nps

# Test case that should work
s = 'hello'
arr = np.array([s])

# This should return 'hello' (full string from index 0 to end)
result = nps.slice(arr, 0, None)

print(f'Input string: {repr(s)}')
print(f'Expected (s[0:None]): {repr(s[0:None])}')
print(f'Got (nps.slice(arr, 0, None)[0]): {repr(result[0])}')
print()

# Let's also test a few more cases to understand the pattern
print("Additional test cases:")
print(f"nps.slice(arr, 1, None)[0]: {repr(nps.slice(arr, 1, None)[0])} (expected: {repr(s[1:None])})")
print(f"nps.slice(arr, 2, None)[0]: {repr(nps.slice(arr, 2, None)[0])} (expected: {repr(s[2:None])})")
print(f"nps.slice(arr, None, 3)[0]: {repr(nps.slice(arr, None, 3)[0])} (expected: {repr(s[None:3])})")
print(f"nps.slice(arr, None, None)[0]: {repr(nps.slice(arr, None, None)[0])} (expected: {repr(s[None:None])})")
print()

# Assert to show the failure
assert result[0] == s[0:None], f"Expected {repr(s[0:None])}, got {repr(result[0])}"
```

<details>

<summary>
AssertionError: Expected 'hello', got np.str_('')
</summary>
```
Input string: 'hello'
Expected (s[0:None]): 'hello'
Got (nps.slice(arr, 0, None)[0]): np.str_('')

Additional test cases:
nps.slice(arr, 1, None)[0]: np.str_('h') (expected: 'ello')
nps.slice(arr, 2, None)[0]: np.str_('he') (expected: 'llo')
nps.slice(arr, None, 3)[0]: np.str_('hel') (expected: 'hel')
nps.slice(arr, None, None)[0]: np.str_('hello') (expected: 'hello')

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/2/repo.py", line 25, in <module>
    assert result[0] == s[0:None], f"Expected {repr(s[0:None])}, got {repr(result[0])}"
           ^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 'hello', got np.str_('')
```
</details>

## Why This Is A Bug

The function's docstring explicitly states it should behave "Like in the regular Python `slice` object". In Python's slice semantics, `stop=None` means "slice to the end of the sequence". This is fundamental Python behavior that users rely on:

- `s[0:None]` equals `s[0:]` which returns the full string from index 0
- `s[1:None]` equals `s[1:]` which returns from index 1 to the end

However, `numpy.strings.slice` violates this contract:
- `nps.slice(arr, 0, None)` returns an empty string instead of 'hello'
- `nps.slice(arr, 1, None)` returns 'h' (one character) instead of 'ello'
- `nps.slice(arr, 2, None)` returns 'he' (two characters) instead of 'llo'

The pattern reveals that when `stop=None` with an explicit start value, the function appears to be treating the start parameter as a count of characters to return from the beginning, rather than as the starting index for the slice.

Interestingly, the function handles None correctly in other scenarios:
- `nps.slice(arr, None, None)` correctly returns the full string
- `nps.slice(arr, None, 3)` correctly returns the first 3 characters

## Relevant Context

The numpy.strings.slice documentation includes this example demonstrating that None values should be supported:
```python
>>> np.strings.slice(b, None, None, -1)
```

This confirms that None parameters are intended functionality, not an edge case. The documentation also explicitly states the function should work "Like in the regular Python `slice` object", making this a clear contract violation.

The bug specifically manifests when:
1. `start` is a non-None integer value
2. `stop` is explicitly None
3. The expected behavior would be to slice from `start` to the end of the string

Documentation link: https://numpy.org/doc/stable/reference/generated/numpy.strings.slice.html

## Proposed Fix

The implementation needs to properly handle the case where `stop=None` with a non-None start. The function should interpret `None` as "slice to the end" consistent with Python's standard slice behavior. A high-level fix would involve:

1. Check if `stop is None` when `start` is not None
2. If so, either:
   - Pass the slice operation without a stop parameter (let it default to end)
   - Or replace None with the string length before processing

The key is ensuring that `slice(start, None)` behaves identically to `slice(start)` or `[start:]` in standard Python, returning all characters from the start index to the end of the string.