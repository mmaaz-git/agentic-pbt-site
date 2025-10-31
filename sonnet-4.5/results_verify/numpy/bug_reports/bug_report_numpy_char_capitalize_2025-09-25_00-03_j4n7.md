# Bug Report: numpy.char.capitalize Truncates Unicode Characters That Expand During Case Conversion

**Target**: `numpy.char.capitalize`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.capitalize` silently truncates results when Unicode case conversion expands the character count, producing incorrect results without warning.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_capitalize_unicode(strings):
    arr = np.array(strings, dtype=str)
    result = char.capitalize(arr)

    for i in range(len(strings)):
        assert result[i] == strings[i].capitalize()
```

**Failing input**: `strings=['ŉabc']`

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

arr = np.array(['ŉabc'], dtype=str)
result = char.capitalize(arr)

print(f"Result: {result[0]!r}")
print(f"Expected: {'ŉabc'.capitalize()!r}")
assert result[0] == 'ʼNabc'
```

## Why This Is A Bug

The function claims to call `str.capitalize` element-wise. The character 'ŉ' (U+0149) capitalizes to 'ʼN' (2 characters). When capitalizing 'ŉabc', Python produces 'ʼNabc' (5 characters), but numpy.char.capitalize produces 'ʼNab' (4 characters) because the input dtype `<U4` cannot hold the extra character. The result is silently truncated, corrupting the data.

## Fix

The `capitalize` function should calculate the maximum possible output length after case conversion and allocate an appropriately sized output array, similar to how `add` and `multiply` handle dtype sizing.