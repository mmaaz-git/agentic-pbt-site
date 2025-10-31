# Bug Report: pandas.core.strings count() Missing regex Parameter

**Target**: `pandas.core.strings.accessor.StringMethods.count`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `count()` method always treats patterns as regex but lacks a `regex=False` parameter, making it impossible to count literal regex metacharacters. This is inconsistent with `replace()` which provides `regex=False`.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
import pandas as pd
import re
import pytest

@given(st.lists(st.text(), min_size=1), st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789', min_size=1), st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789'))
@settings(max_examples=500)
def test_replace_removes_all_occurrences(strings, old, new):
    s = pd.Series(strings)
    count_before = s.str.count(old)
    replaced = s.str.replace(old, new, regex=False)
    count_after = replaced.str.count(old)

    for i in range(len(s)):
        if pd.notna(count_before.iloc[i]) and pd.notna(count_after.iloc[i]):
            assert count_after.iloc[i] == 0
```

**Failing input**: When testing with regex metacharacters like ')', '(', '*', etc., the test crashes before even getting to the assertion.

## Reproducing the Bug

```python
import pandas as pd

s = pd.Series(['test)test', 'hello(world', 'dot.here'])

s.str.count(')')

s.str.count('(')

s.str.count('.')

s.str.count(')', regex=False)

print("Expected (Python str.count):")
print(f"'test)test'.count(')'): {s.iloc[0].count(')')}")
print(f"'hello(world'.count('('): {s.iloc[1].count('(')}")
print(f"'dot.here'.count('.'): {s.iloc[2].count('.')}")
```

**Output:**
```
PatternError: unbalanced parenthesis at position 0
PatternError: missing ), unterminated subpattern at position 0
[9, 11, 8]  # Wrong! Should be [1, 1, 1]
TypeError: StringMethods.count() got an unexpected keyword argument 'regex'

Expected (Python str.count):
'test)test'.count(')'): 1
'hello(world'.count('('): 1
'dot.here'.count('.'): 1
```

## Why This Is A Bug

1. **API Inconsistency**: `str.replace()` has a `regex` parameter (default `False` in recent pandas), but `str.count()` does not
2. **Contract Violation**: The method signature `count(pat, flags=0)` and documentation describe `pat` as a "Valid regular expression", but users expect it to work like Python's `str.count()` for literal strings
3. **User Impact**: Cannot count occurrences of common characters like parentheses without escaping them manually with `re.escape()`
4. **Silent Incorrect Results**: For metacharacters like '.', the method silently returns wrong results (counts all characters instead of literal dots)

## Fix

Add a `regex` parameter to `count()` to match the API of `replace()` and other string methods:

```diff
--- a/pandas/core/strings/accessor.py
+++ b/pandas/core/strings/accessor.py
@@ -2434,7 +2434,7 @@ class StringMethods(NoNewAttributesMixin):
         )

     @forbid_nonstring_types(["bytes", "mixed", "mixed-integer"])
-    def count(self, pat, flags: int = 0):
+    def count(self, pat, flags: int = 0, regex: bool = True):
         """
         Count occurrences of pattern in each string of the Series/Index.

@@ -2444,6 +2444,9 @@ class StringMethods(NoNewAttributesMixin):
         Parameters
         ----------
         pat : str
-            Valid regular expression.
+            String to match. Can be a regular expression if regex=True.
+        regex : bool, default True
+            If True, assumes the pattern is a regular expression.
+            If False, treats the pattern as a literal string.
         flags : int, default 0, meaning no flags
             Flags for the `re` module. For a complete list, `see here
             <https://docs.python.org/3/howto/regex.html#compilation-flags>`_.
--- a/pandas/core/strings/object_array.py
+++ b/pandas/core/strings/object_array.py
@@ -109,8 +109,11 @@ class ObjectStringArrayMixin(BaseStringArrayMethods):
-    def _str_count(self, pat, flags=0):
-        regex = re.compile(pat, flags=flags)
+    def _str_count(self, pat, flags=0, regex=True):
+        if regex:
+            pattern = re.compile(pat, flags=flags)
+        else:
+            pattern = re.compile(re.escape(pat), flags=flags)
         f = lambda x: len(pattern.findall(x))
         return self._str_map(f, dtype="int64")
```