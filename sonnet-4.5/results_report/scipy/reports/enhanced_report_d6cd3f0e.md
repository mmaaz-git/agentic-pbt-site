# Bug Report: scipy.io.matlab Character Array Dimension Growth on Round-Trip

**Target**: `scipy.io.matlab.loadmat` / `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Character arrays loaded from MATLAB files with `chars_as_strings=False` gain an extra trailing dimension on each save-load cycle, violating the idempotent round-trip property that the documentation promises.

## Property-Based Test

```python
import scipy.io.matlab as mat
import numpy as np
import tempfile
import os
from hypothesis import given, settings, strategies as st


@settings(max_examples=100)
@given(st.booleans())
def test_chars_as_strings_roundtrip(chars_as_strings):
    test_dict = {'text': 'hello world'}

    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        temp_file1 = f.name
    with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
        temp_file2 = f.name

    try:
        mat.savemat(temp_file1, test_dict)
        loaded1 = mat.loadmat(temp_file1, chars_as_strings=chars_as_strings)

        user_keys1 = {k: v for k, v in loaded1.items() if not k.startswith('__')}
        mat.savemat(temp_file2, user_keys1)
        loaded2 = mat.loadmat(temp_file2, chars_as_strings=chars_as_strings)

        user_keys2 = {k: v for k, v in loaded2.items() if not k.startswith('__')}

        for key in user_keys1:
            assert key in user_keys2
            assert np.array_equal(user_keys1[key], user_keys2[key]), \
                f"Arrays not equal for key '{key}': shape {user_keys1[key].shape} vs {user_keys2[key].shape}"

    finally:
        if os.path.exists(temp_file1):
            os.remove(temp_file1)
        if os.path.exists(temp_file2):
            os.remove(temp_file2)


if __name__ == "__main__":
    test_chars_as_strings_roundtrip()
```

<details>

<summary>
**Failing input**: `chars_as_strings=False`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 41, in <module>
    test_chars_as_strings_roundtrip()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 9, in test_chars_as_strings_roundtrip
    @given(st.booleans())
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/28/hypo.py", line 30, in test_chars_as_strings_roundtrip
    assert np.array_equal(user_keys1[key], user_keys2[key]), \
           ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Arrays not equal for key 'text': shape (1, 11) vs (1, 11, 1)
Falsifying example: test_chars_as_strings_roundtrip(
    chars_as_strings=False,
)
```
</details>

## Reproducing the Bug

```python
import scipy.io.matlab as mat
import numpy as np
import tempfile
import os

text = 'hello'
test_dict = {'text': text}

with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    temp1 = f.name
with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    temp2 = f.name
with tempfile.NamedTemporaryFile(suffix='.mat', delete=False) as f:
    temp3 = f.name

try:
    # First save and load
    mat.savemat(temp1, test_dict)
    loaded1 = mat.loadmat(temp1, chars_as_strings=False)
    print(f"1st load shape: {loaded1['text'].shape}")
    print(f"1st load type: {type(loaded1['text'])}")
    print(f"1st load content: {loaded1['text']}")
    print()

    # Second save and load
    user_data1 = {k: v for k, v in loaded1.items() if not k.startswith('__')}
    mat.savemat(temp2, user_data1)
    loaded2 = mat.loadmat(temp2, chars_as_strings=False)
    print(f"2nd load shape: {loaded2['text'].shape}")
    print(f"2nd load type: {type(loaded2['text'])}")
    print(f"2nd load content shape details: {loaded2['text']}")
    print()

    # Third save and load
    user_data2 = {k: v for k, v in loaded2.items() if not k.startswith('__')}
    mat.savemat(temp3, user_data2)
    loaded3 = mat.loadmat(temp3, chars_as_strings=False)
    print(f"3rd load shape: {loaded3['text'].shape}")
    print(f"3rd load type: {type(loaded3['text'])}")
    print(f"3rd load content shape details: {loaded3['text']}")

    # Verify arrays are not equal after round-trip
    print(f"\nArrays equal after 1st and 2nd load? {np.array_equal(loaded1['text'], loaded2['text'])}")

finally:
    os.remove(temp1)
    os.remove(temp2)
    os.remove(temp3)
```

<details>

<summary>
Arrays progressively gain dimensions with each round-trip
</summary>
```
1st load shape: (1, 5)
1st load type: <class 'numpy.ndarray'>
1st load content: [['h' 'e' 'l' 'l' 'o']]

2nd load shape: (1, 5, 1)
2nd load type: <class 'numpy.ndarray'>
2nd load content shape details: [[['h']
  ['e']
  ['l']
  ['l']
  ['o']]]

3rd load shape: (1, 5, 1, 1)
3rd load type: <class 'numpy.ndarray'>
3rd load content shape details: [[[['h']]

  [['e']]

  [['l']]

  [['l']]

  [['o']]]]

Arrays equal after 1st and 2nd load? False
```
</details>

## Why This Is A Bug

This behavior violates the documented promise of round-trip support in multiple ways:

1. **Documentation Promise Violation**: The `loadmat` function documentation explicitly states that `struct_as_record=True` (the default) is used "because it allows easier round-trip load and save of MATLAB files." This promise extends to the entire function, not just one parameter.

2. **Non-Idempotent Behavior**: After the first load-save cycle, subsequent operations should be idempotent - saving and loading the same data should not change its structure. However, with `chars_as_strings=False`, each cycle adds a new trailing dimension indefinitely:
   - Original: Python string 'hello'
   - 1st load: numpy array shape `(1, 5)`
   - 2nd load: numpy array shape `(1, 5, 1)`
   - 3rd load: numpy array shape `(1, 5, 1, 1)`
   - nth load: numpy array shape `(1, 5, 1, 1, ..., 1)` with n-2 trailing 1s

3. **MATLAB Compatibility Issue**: The `matlab_compatible` parameter sets `chars_as_strings=False` to match MATLAB's behavior. This bug means MATLAB compatibility mode produces progressively corrupted data structures.

4. **Data Structure Corruption**: While the character content is preserved, the array structure changes unexpectedly. This breaks any code that depends on consistent array shapes, causing downstream failures in data processing pipelines.

5. **Silent Failure**: The bug occurs silently without warnings or errors, making it difficult to detect until array shape mismatches cause failures elsewhere in the code.

## Relevant Context

The bug appears to stem from how scipy handles character arrays during the conversion process between Python strings, numpy arrays, and MATLAB's internal representation:

- MATLAB stores character arrays as 2D arrays where each row is a string
- When `chars_as_strings=False`, scipy converts Python strings to numpy character arrays
- On each save, scipy appears to add an extra dimension to ensure MATLAB compatibility
- On each load, this extra dimension is preserved rather than being normalized

The issue is in scipy version 1.16.2 but likely affects other versions as well. The bug only manifests when:
- Using `chars_as_strings=False` (not the default)
- Performing multiple round-trip save/load operations
- Working with character/string data (not numeric data)

Documentation references:
- scipy.io.loadmat: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html
- scipy.io.savemat: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.savemat.html

## Proposed Fix

The fix requires modifying how scipy handles character arrays that have already been loaded from MATLAB files. The savemat function should detect when a character array already has the expected MATLAB format and avoid adding additional dimensions. Here's a conceptual approach:

In the savemat function's character array handling logic, check if the input array:
1. Is already a numpy array (not a Python string)
2. Has dtype of '<U1' or similar character type
3. Already has 2 or more dimensions

If all conditions are met, the array likely came from a previous loadmat and should be saved without further reshaping. The exact implementation would need to be in the scipy.io.matlab module's internal conversion functions, likely in the code that handles the `chars_as_strings` parameter during the save process.