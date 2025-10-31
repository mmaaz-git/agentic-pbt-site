# Bug Report: numpy.char.upper Truncates Unicode Characters That Expand When Uppercased

**Target**: `numpy.char.upper`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.upper` silently truncates results when Unicode case conversion expands the character count, producing incorrect results without warning.

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
```

**Failing input**: `strings=['ß']`

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

arr = np.array(['ß'], dtype=str)
result = char.upper(arr)

print(f"Result: {result[0]!r}")
print(f"Expected: {'ß'.upper()!r}")
assert result[0] == 'SS'
```

## Why This Is A Bug

The function claims to call `str.upper` element-wise. In Unicode, several characters expand when converted to uppercase (e.g., German 'ß' → 'SS', ligature 'ﬁ' → 'FI'). The input array has dtype `<U1` (1 character), but the output should be 2 characters. Instead of resizing the output dtype, numpy.char.upper truncates to 'S', silently corrupting the data.

## Fix

The `upper` function should calculate the maximum possible output length for each string after case conversion and allocate an appropriately sized output array, similar to how `add` and `multiply` handle dtype sizing.