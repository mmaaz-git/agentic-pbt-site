# Bug Report: numpy.char Case Functions Don't Expand Characters Like Python

**Target**: `numpy.char.upper`, `numpy.char.capitalize`, `numpy.char.swapcase`, `numpy.char.title`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Multiple numpy.char case conversion functions (upper, capitalize, title, swapcase) claim to call their Python str equivalents element-wise, but fail to properly handle characters that expand to multiple characters during case conversion, such as German ß.

## Property-Based Test

```python
import numpy.char as char
import numpy as np
from hypothesis import given, strategies as st, settings


st_text = st.text(
    alphabet=st.characters(blacklist_categories=('Cs',), blacklist_characters='\x00'),
    min_size=0,
    max_size=20
)


@given(st.lists(st_text, min_size=1, max_size=10).map(lambda lst: np.array(lst, dtype='U')))
@settings(max_examples=1000)
def test_upper_matches_python(arr):
    result = char.upper(arr)
    for i in range(len(arr)):
        assert result[i] == arr[i].upper()
```

**Failing input**: `array(['ß'])`

## Reproducing the Bug

```python
import numpy.char as char
import numpy as np

test_char = 'ß'
arr = np.array([test_char])

print(f'Testing: {test_char!r}')
print(f'upper:      numpy={char.upper(arr)[0]!r:5} Python={test_char.upper()!r}')
print(f'capitalize: numpy={char.capitalize(arr)[0]!r:5} Python={test_char.capitalize()!r}')
print(f'title:      numpy={char.title(arr)[0]!r:5} Python={test_char.title()!r}')
print(f'swapcase:   numpy={char.swapcase(arr)[0]!r:5} Python={test_char.swapcase()!r}')
```

Output:
```
Testing: 'ß'
upper:      numpy='S'   Python='SS'
capitalize: numpy='S'   Python='Ss'
title:      numpy='S'   Python='Ss'
swapcase:   numpy='S'   Python='SS'
```

## Why This Is A Bug

All four affected functions have documentation claiming they call Python's str methods element-wise:
- `upper`: "Calls :meth:`str.upper` element-wise"
- `capitalize`: "Calls :meth:`str.capitalize` element-wise"
- `title`: "Calls :meth:`str.title` element-wise"
- `swapcase`: "Calls :meth:`str.swapcase` element-wise"

However, they produce different results than Python for characters with complex case mappings:
- German ß (U+00DF) uppercases to 'SS' in Python, but only 'S' in numpy
- Ligatures like ﬁ (U+FB01) titlecase to 'Fi' in Python, but only 'F' in numpy

This is a contract violation - the documented behavior does not match actual behavior. The root cause is that numpy's implementation cannot handle case conversions where the result has a different number of characters than the input.

## Fix

The numpy implementation needs to properly handle characters that expand during case conversion. The current implementation appears to perform character-by-character mapping with a fixed-size output buffer, preventing expansion.

Potential fixes:
1. Use Python's actual str methods with proper output sizing
2. Update documentation to clarify limitations
3. Implement proper Unicode case mapping with character expansion

Suggested approach:
```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -xxx,x +xxx,x @@
 def upper(a):
-    return _vec_string(a, a.dtype, 'upper')
+    # Use Python's str.upper to properly handle character expansion
+    def _upper_with_expansion(s):
+        result = s.upper()
+        return result
+    return np.vectorize(_upper_with_expansion, otypes=[object])(a).astype(str)
```

Similar changes would be needed for capitalize, title, and swapcase. The challenge is handling the dtype change when characters expand, which may require returning object arrays or dynamically sizing the output dtype.