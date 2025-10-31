# Bug Report: scipy.io.matlab Character Array Round-Trip

**Target**: `scipy.io.matlab.loadmat` / `scipy.io.matlab.savemat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Character arrays loaded with `chars_as_strings=False` gain an extra dimension on each save-load cycle, violating the idempotent round-trip property.

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
            assert np.array_equal(user_keys1[key], user_keys2[key])

    finally:
        if os.path.exists(temp_file1):
            os.remove(temp_file1)
        if os.path.exists(temp_file2):
            os.remove(temp_file2)
```

**Failing input**: `chars_as_strings=False`

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
    mat.savemat(temp1, test_dict)
    loaded1 = mat.loadmat(temp1, chars_as_strings=False)
    print(f"1st load shape: {loaded1['text'].shape}")

    user_data1 = {k: v for k, v in loaded1.items() if not k.startswith('__')}
    mat.savemat(temp2, user_data1)
    loaded2 = mat.loadmat(temp2, chars_as_strings=False)
    print(f"2nd load shape: {loaded2['text'].shape}")

    user_data2 = {k: v for k, v in loaded2.items() if not k.startswith('__')}
    mat.savemat(temp3, user_data2)
    loaded3 = mat.loadmat(temp3, chars_as_strings=False)
    print(f"3rd load shape: {loaded3['text'].shape}")

finally:
    os.remove(temp1)
    os.remove(temp2)
    os.remove(temp3)
```

Output:
```
1st load shape: (1, 5)
2nd load shape: (1, 5, 1)
3rd load shape: (1, 5, 1, 1)
```

## Why This Is A Bug

The `loadmat` docstring explicitly states that the default `struct_as_record=True` setting is used "because it allows easier round-trip load and save of MATLAB files." This implies that round-trip operations should be idempotent after the first cycle.

However, when loading character arrays with `chars_as_strings=False`, each save-load cycle adds a new trailing dimension:
- Original string → 1st load: `(1, 5)`
- 2nd load: `(1, 5, 1)`
- 3rd load: `(1, 5, 1, 1)`
- And so on...

This violates the idempotent round-trip property and will cause issues for users who:
1. Load MATLAB files with character arrays
2. Perform operations on the data
3. Save and reload the data
4. Find that the array shape has unexpectedly changed

## Fix

The issue likely stems from how character arrays are being reshaped during the save or load process. The character array should maintain a stable shape after the first round-trip. The fix would involve ensuring that when saving character arrays that already have the expected MATLAB format (i.e., already loaded from a .mat file), they are not further reshaped.