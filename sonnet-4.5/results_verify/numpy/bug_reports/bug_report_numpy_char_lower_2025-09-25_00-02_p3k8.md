# Bug Report: numpy.char.lower Truncates Unicode Characters That Expand When Lowercased

**Target**: `numpy.char.lower`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.lower` silently truncates results when Unicode case conversion expands the character count, producing incorrect results without warning.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_upper_lower_unicode(strings):
    arr = np.array(strings, dtype=str)
    lower_result = char.lower(arr)

    for i in range(len(strings)):
        assert lower_result[i] == strings[i].lower()
```

**Failing input**: `strings=['İ']`

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

arr = np.array(['İ'], dtype=str)
result = char.lower(arr)

print(f"Result: {result[0]!r}")
print(f"Expected: {'İ'.lower()!r}")
assert result[0] == 'i̇'
```

## Why This Is A Bug

The function claims to call `str.lower` element-wise. The Turkish capital 'İ' (U+0130) converts to lowercase as 'i̇' (U+0069 + U+0307 combining dot above), which is 2 characters. The input array has dtype `<U1` (1 character). Instead of resizing the output dtype to accommodate the result, numpy.char.lower truncates to 'i', silently corrupting the data.

## Fix

The `lower` function should calculate the maximum possible output length for each string after case conversion and allocate an appropriately sized output array, similar to how `add` and `multiply` handle dtype sizing.