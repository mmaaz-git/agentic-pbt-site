# Bug Report: Cython.Build.Dependencies.parse_list Hash Character Corruption

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_list()` function incorrectly treats `#` as a comment delimiter in compiler directive values, corrupting any value containing a hash character.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet=st.characters(blacklist_categories=('Cs',)), min_size=1)))
def test_parse_list_bracket_delimited(items):
    assume(all(item.strip() for item in items))
    assume(all(',' not in item and '"' not in item and "'" not in item for item in items))
    s = '[' + ', '.join(items) + ']'
    result = parse_list(s)
    assert result == items
```

**Failing input**: `items=['#']`

## Reproducing the Bug

```python
from Cython.Build.Dependencies import parse_list

result = parse_list('[foo#bar]')
print(result)

assert result == ['foo#bar'], f"Expected ['foo#bar'], got {result}"
```

Expected: `['foo#bar']`
Actual: `['foo#__Pyx_L1_']`

Additional examples:
- `parse_list('[libA, libB#version]')` returns `['libA', 'libB#__Pyx_L1_']` instead of `['libA', 'libB#version']`
- `parse_list('foo#bar baz')` returns `['foo#__Pyx_L1_']` instead of `['foo#bar', 'baz']`

## Why This Is A Bug

The `parse_list()` function is used to parse values from Cython compiler directives (lines 197-198 in Dependencies.py):

```python
if type in (list, transitive_list):
    value = parse_list(value)
```

These directive values come from comments like:
```python
# distutils: libraries = lib#version
```

The value string `'lib#version'` is already extracted from the directive line and should be treated as a simple list value, not as Python source code. However, `parse_list()` calls `strip_string_literals()` which is designed for parsing Python source code and treats `#` as starting a comment.

This corrupts legitimate values that contain `#` characters (e.g., version tags, file hashes, or preprocessor-style names like `DEBUG#2`).

## Fix

The issue is at line 128 of Dependencies.py. The function should not treat `#` as a comment delimiter when parsing simple list values. A proper fix would be to only handle quoted strings without full Python comment parsing:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -125,7 +125,14 @@ def parse_list(s):
         s = s[1:-1]
         delimiter = ','
     else:
         delimiter = ' '
-    s, literals = strip_string_literals(s)
+
+    literals = {}
+    counter = 0
+    def replace_quotes(m):
+        nonlocal counter
+        counter += 1
+        label = f"__Pyx_L{counter}_"
+        literals[label] = m.group(1)
+        return label
+    s = re.sub(r'''(["'])(?:(?=(\\?))\2.)*?\1''', replace_quotes, s)
+
     def unquote(literal):
         literal = literal.strip()
         if literal[0] in "'\"":
```

Alternatively, a simpler fix is to avoid calling `strip_string_literals()` entirely and use a more targeted approach for handling quoted strings in list values.