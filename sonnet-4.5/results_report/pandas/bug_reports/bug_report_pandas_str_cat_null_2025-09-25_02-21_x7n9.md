# Bug Report: pandas.core.strings Null Character Separator

**Target**: `pandas.core.strings.accessor.cat_core` and `pd.Series.str.cat`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `cat_core` function (used by `Series.str.cat()`) silently drops null characters (`\x00`) when used as separators, resulting in incorrect concatenation.

## Property-Based Test

```python
import pandas as pd
from hypothesis import given, strategies as st, settings, example


@settings(max_examples=1000)
@given(
    st.lists(st.text(), min_size=1, max_size=10),
    st.lists(st.text(), min_size=1, max_size=10),
    st.text(alphabet=st.characters(min_codepoint=0, max_codepoint=127), max_size=3)
)
@example(['a'], ['b'], '\x00')
@example([''], [''], '\x00')
def test_str_cat_separator_property(strings1, strings2, sep):
    min_len = min(len(strings1), len(strings2))
    s1 = pd.Series(strings1[:min_len])
    s2 = pd.Series(strings2[:min_len])

    result = s1.str.cat(s2, sep=sep)

    for i in range(min_len):
        expected = strings1[i] + sep + strings2[i]
        assert result.iloc[i] == expected
```

**Failing inputs**:
- `strings1=['a'], strings2=['b'], sep='\x00'` → expected `'a\x00b'`, got `'ab'`
- `strings1=[''], strings2=[''], sep='\x00'` → expected `'\x00'`, got `''`

## Reproducing the Bug

```python
import pandas as pd

s1 = pd.Series(['a', ''])
s2 = pd.Series(['b', ''])

result = s1.str.cat(s2, sep='\x00')
expected = ['a\x00b', '\x00']

assert result.tolist() == expected
```

Output:
```
AssertionError: assert ['ab', ''] == ['a\x00b', '\x00']
```

## Why This Is A Bug

The `cat_core` function uses NumPy's array addition to concatenate strings. NumPy silently drops null characters (`\x00`) when adding them to object-dtype string arrays:

```python
import numpy as np
arr = np.array(['hello'], dtype=object)
result = arr + '\x00'
```

This violates the documented behavior of `Series.str.cat()`, which should concatenate strings with the given separator. Null-delimited formats are used in binary protocols and certain data formats, so this causes silent data corruption.

## Fix

Pandas can work around NumPy's null-character handling by using Python string operations:

```diff
--- a/pandas/core/strings/accessor.py
+++ b/pandas/core/strings/accessor.py
@@ -3489,8 +3489,11 @@ def cat_core(list_of_columns: list, sep: str):
     """
     if sep == "":
         arr_of_cols = np.asarray(list_of_columns, dtype=object)
         return np.sum(arr_of_cols, axis=0)
-    list_with_sep = [sep] * (2 * len(list_of_columns) - 1)
-    list_with_sep[::2] = list_of_columns
-    arr_with_sep = np.asarray(list_with_sep, dtype=object)
-    return np.sum(arr_with_sep, axis=0)
+
+    n_elements = len(list_of_columns[0])
+    result = np.empty(n_elements, dtype=object)
+    for i in range(n_elements):
+        result[i] = sep.join(col[i] for col in list_of_columns)
+    return result
```