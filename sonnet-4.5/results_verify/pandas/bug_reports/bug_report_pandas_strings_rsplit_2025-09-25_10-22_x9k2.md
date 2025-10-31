# Bug Report: pandas.core.strings rsplit() Missing regex Parameter

**Target**: `pandas.core.strings.accessor.StringMethods.rsplit`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `rsplit()` method lacks a `regex` parameter while its counterpart `split()` has one, creating API inconsistency between two related methods.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd

@given(st.lists(st.text(), min_size=1), st.text(min_size=1))
@settings(max_examples=500)
def test_split_rsplit_api_parity(strings, sep):
    s = pd.Series(strings)

    split_literal = s.str.split(sep, regex=False)

    rsplit_literal = s.str.rsplit(sep, regex=False)

    for i in range(len(s)):
        if pd.notna(s.iloc[i]):
            assert len(split_literal.iloc[i]) == len(rsplit_literal.iloc[i])
```

**Failing input**: Any input causes TypeError when trying to use `regex=False` with `rsplit()`.

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series(['a-b-c'])

split_result = s.str.split('-', regex=False)
print(f"split('-', regex=False): {split_result.iloc[0]}")

rsplit_result = s.str.rsplit('-', regex=False)
print(f"rsplit('-', regex=False): {rsplit_result.iloc[0]}")
```

**Output:**
```
split('-', regex=False): ['a', 'b', 'c']
TypeError: StringMethods.rsplit() got an unexpected keyword argument 'regex'
```

## Why This Is A Bug

1. **API Inconsistency**: `split()` has a `regex` parameter but `rsplit()` does not
2. **User Expectation**: Users expect symmetric APIs between `split()` and `rsplit()`
3. **Documentation Gap**: The lack of parity is not clearly documented

Note: `rsplit()` currently always uses literal string matching (via Python's `str.rsplit()`), which is the desired behavior when `regex=False`. However, having an explicit parameter would improve API consistency and clarity.

## Fix

Add a `regex` parameter to `rsplit()` to match `split()`'s API:

```diff
--- a/pandas/core/strings/accessor.py
+++ b/pandas/core/strings/accessor.py
@@ -950,7 +950,7 @@ class StringMethods(NoNewAttributesMixin):
         return new_series

     @forbid_nonstring_types(["bytes", "mixed", "mixed-integer"])
-    def rsplit(self, pat=None, *, n=-1, expand: bool = False):
+    def rsplit(self, pat=None, *, n=-1, expand: bool = False, regex: bool = False):
         """
         Split strings around given separator/delimiter.

@@ -962,6 +962,9 @@ class StringMethods(NoNewAttributesMixin):
         pat : str, optional
             String to split on. If not specified, split on whitespace.
+        regex : bool, default False
+            If True, assumes the pattern is a regular expression.
+            If False, treats the pattern as a literal string.
         n : int, default -1 (all)
             Limit number of splits in output.
             ``None``, 0 and -1 will be interpreted as return all splits.
@@ -1002,7 +1005,7 @@ class StringMethods(NoNewAttributesMixin):
         [Index(['a', 'b'], dtype='object'), Index(['c', 'd'], dtype='object')]
         """
         func = self._orig if isinstance(self._orig, ABCIndex) else lambda x: x
-        result = self._data.array._str_rsplit(pat, n=n)
+        result = self._data.array._str_rsplit(pat, n=n, regex=regex)
         return self._expand_single_result_series(result, expand, func)
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -400,8 +400,20 @@ class ObjectStringArrayMixin(BaseStringArrayMethods):
         return self._str_map(f, dtype=object)

-    def _str_rsplit(self, pat=None, n=-1):
+    def _str_rsplit(self, pat=None, n=-1, regex=False):
+        if pat is None:
+            if n is None or n == 0:
+                n = -1
+            f = lambda x: x.rsplit(pat, n)
+        elif regex:
+            pattern = re.compile(pat) if isinstance(pat, str) else pat
+            if n is None or n == -1:
+                n = 0
+            f = lambda x: list(reversed(pattern.split(x[::-1], maxsplit=n)))
+        else:
+            if n is None or n == 0:
+                n = -1
+            f = lambda x: x.rsplit(pat, n)
-        if n is None or n == 0:
-            n = -1
-        f = lambda x: x.rsplit(pat, n)
         return self._str_map(f, dtype="object")
```