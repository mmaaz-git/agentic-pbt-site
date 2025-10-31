# Bug Report: numpy.char.swapcase Truncates Unicode Characters That Expand During Case Conversion

**Target**: `numpy.char.swapcase`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.swapcase` silently truncates results when Unicode case conversion expands the character count, producing incorrect results without warning.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_swapcase_unicode(strings):
    arr = np.array(strings, dtype=str)
    result = char.swapcase(arr)

    for i in range(len(strings)):
        assert result[i] == strings[i].swapcase()
```

**Failing input**: `strings=['ß']`

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

arr = np.array(['ß'], dtype=str)
result = char.swapcase(arr)

print(f"Result: {result[0]!r}")
print(f"Expected: {'ß'.swapcase()!r}")
assert result[0] == 'SS'
```

## Why This Is A Bug

The function claims to call `str.swapcase` element-wise. German lowercase 'ß' swaps to uppercase 'SS' (2 characters). With input dtype `<U1` (1 character), the output should expand to 2 characters. Instead of resizing the output dtype, numpy.char.swapcase truncates to 'S', silently corrupting the data.

## Fix

The `swapcase` function should calculate the maximum possible output length after case conversion and allocate an appropriately sized output array, similar to how `add` and `multiply` handle dtype sizing.