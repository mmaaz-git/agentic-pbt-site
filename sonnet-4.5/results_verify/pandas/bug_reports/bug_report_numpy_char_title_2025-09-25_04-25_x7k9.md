# Bug Report: numpy.char.title Does Not Match Python str.title for Ligatures

**Target**: `numpy.char.title`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

`numpy.char.title()` claims to call `str.title` element-wise (per its documentation), but produces different results than Python's `str.title()` for ligatures and certain special characters like ß.

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
def test_title_matches_python(arr):
    result = char.title(arr)
    for i in range(len(arr)):
        numpy_result = result[i]
        python_result = arr[i].title()
        assert numpy_result == python_result
```

**Failing input**: `array(['ß'])`

## Reproducing the Bug

```python
import numpy.char as char
import numpy as np

test_cases = [
    ('ß', 'Ss'),
    ('ﬁ', 'Fi'),
    ('ﬂ', 'Fl'),
    ('ﬀ', 'Ff'),
]

for s, expected_python in test_cases:
    arr = np.array([s])
    numpy_result = char.title(arr)[0]
    python_result = s.title()

    print(f'{s!r}: numpy={numpy_result!r}, Python={python_result!r}, match={numpy_result == python_result}')

arr = np.array(['ß'])
print(f'\nchar.title(["ß"]) = {char.title(arr)[0]!r} (expected: "Ss")')

arr2 = np.array(['ﬁ'])
print(f'char.title(["ﬁ"]) = {char.title(arr2)[0]!r} (expected: "Fi")')
```

Output:
```
'ß': numpy='S', Python='Ss', match=False
'ﬁ': numpy='F', Python='Fi', match=False
'ﬂ': numpy='F', Python='Fl', match=False
'ﬀ': numpy='F', Python='Ff', match=False

char.title(["ß"]) = 'S' (expected: "Ss")
char.title(["ﬁ"]) = 'F' (expected: "Fi")
```

## Why This Is A Bug

The `numpy.char.title` function's documentation explicitly states: "Calls :meth:`str.title` element-wise."

However, the actual behavior differs from Python's `str.title()` for certain Unicode characters:
- **German ß** (LATIN SMALL LETTER SHARP S): Python expands to 'Ss', numpy gives 'S'
- **Ligatures** (ﬁ, ﬂ, ﬀ, ﬆ): Python expands to multi-character sequences, numpy gives single characters

This is a contract violation - the documented behavior does not match the actual behavior.

## Fix

The numpy implementation needs to either:
1. Actually call Python's `str.title()` on each element, or
2. Update the documentation to clarify that it uses a different titlecasing algorithm

Option 1 (match the documented behavior):
```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -xxx,x +xxx,x @@
 def title(a):
-    return _vec_string(a, a.dtype, 'title')
+    # Use Python's str.title to match documentation
+    return np.vectorize(lambda s: s.title(), otypes=[a.dtype])(a)
```

Option 2 (document the actual behavior):
```diff
--- a/numpy/_core/strings.py
+++ b/numpy/_core/strings.py
@@ -xxx,x +xxx,x @@
 """
 Return element-wise title cased version of string or unicode.

-Calls :meth:`str.title` element-wise.
+Applies title casing element-wise using Unicode title case mapping.
+Note: Results may differ from Python's str.title() for ligatures
+and special characters.
```

The preferred fix is Option 1 to maintain compatibility with Python's string behavior.