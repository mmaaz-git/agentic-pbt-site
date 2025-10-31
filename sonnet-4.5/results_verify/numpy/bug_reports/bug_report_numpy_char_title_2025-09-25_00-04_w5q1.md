# Bug Report: numpy.char.title Truncates Unicode Characters That Expand During Case Conversion

**Target**: `numpy.char.title`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

`numpy.char.title` silently truncates results when Unicode case conversion expands the character count, producing incorrect results without warning.

## Property-Based Test

```python
import numpy as np
import numpy.char as char
from hypothesis import given, strategies as st

@given(st.lists(st.text(alphabet=st.characters(min_codepoint=0x100, max_codepoint=0x1000, blacklist_categories=('Cs',)), min_size=1, max_size=10), min_size=1, max_size=10))
def test_title_unicode(strings):
    arr = np.array(strings, dtype=str)
    result = char.title(arr)

    for i in range(len(strings)):
        assert result[i] == strings[i].title()
```

**Failing input**: `strings=['ﬁ test']`

## Reproducing the Bug

```python
import numpy as np
import numpy.char as char

arr = np.array(['ﬁ test'], dtype=str)
result = char.title(arr)

print(f"Result: {result[0]!r}")
print(f"Expected: {'ﬁ test'.title()!r}")
assert result[0] == 'Fi Test'
```

## Why This Is A Bug

The function claims to call `str.title` element-wise. The ligature 'ﬁ' (U+FB01) title-cases to 'Fi' (2 characters). When title-casing 'ﬁ test', Python produces 'Fi Test' (7 characters), but numpy.char.title produces 'Fi Tes' (6 characters) because the input dtype `<U7` cannot accommodate the expansion. The result is silently truncated, corrupting the data.

## Fix

The `title` function should calculate the maximum possible output length after case conversion and allocate an appropriately sized output array, similar to how `add` and `multiply` handle dtype sizing.