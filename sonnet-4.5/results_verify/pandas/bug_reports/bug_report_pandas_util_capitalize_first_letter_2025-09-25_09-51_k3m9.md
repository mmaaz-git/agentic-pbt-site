# Bug Report: pandas.util.capitalize_first_letter Unicode Length Change

**Target**: `pandas.util.capitalize_first_letter`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `capitalize_first_letter` function changes string length for certain Unicode characters that expand when uppercased (e.g., German ß → SS).

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.util import capitalize_first_letter
from hypothesis import given, strategies as st

@given(st.text(min_size=1))
def test_rest_unchanged(s):
    result = capitalize_first_letter(s)
    assert result[1:] == s[1:]

@given(st.text(min_size=1))
def test_first_char_uppercase(s):
    result = capitalize_first_letter(s)
    assert result[0] == s[0].upper()
```

**Failing input**: `'ß'`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.util import capitalize_first_letter

s = 'ß'
result = capitalize_first_letter(s)

print(f"Input: {repr(s)} (length {len(s)})")
print(f"Output: {repr(result)} (length {len(result)})")

assert len(result) == 2
assert result == 'SS'
assert s[1:] == ''
assert result[1:] == 'S'
```

## Why This Is A Bug

The function uses `s[:1].upper() + s[1:]` which fails to preserve string length for Unicode characters that expand when uppercased. The German ß uppercases to SS, changing length from 1 to 2. This violates the implicit contract that:
- `result[1:]` should equal `s[1:]` (rest of string unchanged)
- Capitalizing a single character shouldn't change string length

While internal usage in `pandas.core.dtypes.dtypes.PeriodDtype.__eq__` only passes ASCII strings, the function is public in `pandas.util` with no documented restrictions, so users could call it with any Unicode string.

## Fix

Use `.capitalize()` instead of manual slicing, or use `s[0].upper() + s[1:]` if non-empty (though this still has issues with multi-codepoint graphemes). For the specific use case in PeriodDtype, consider using `.lower()` comparison instead:

```diff
--- a/pandas/util/__init__.py
+++ b/pandas/util/__init__.py
@@ -27,5 +27,5 @@ def __getattr__(key: str):


 def capitalize_first_letter(s):
-    return s[:1].upper() + s[1:]
+    return s[0].upper() + s[1:] if s else s
```

Note: This fix still has issues with multi-codepoint characters. A more robust solution for PeriodDtype would be case-insensitive comparison:

```diff
--- a/pandas/core/dtypes/dtypes.py
+++ b/pandas/core/dtypes/dtypes.py
@@ -1088,7 +1088,7 @@ class PeriodDtype(PeriodDtypeBase, PandasExtensionDtype):
     def __eq__(self, other: object) -> bool:
         if isinstance(other, str):
-            return other in [self.name, capitalize_first_letter(self.name)]
+            return other.lower() == self.name.lower()

         return super().__eq__(other)
```