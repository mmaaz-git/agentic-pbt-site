# Bug Report: Cython.Build.Dependencies.parse_list Mangles '#' Characters

**Target**: `Cython.Build.Dependencies.parse_list`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_list` function incorrectly treats `#` as a comment marker, causing it to mangle library names, macro names, and other values that legitimately contain `#` characters.

## Property-Based Test

```python
from hypothesis import given, settings, strategies as st
from Cython.Build.Dependencies import parse_list

@given(st.lists(st.text(alphabet='abcdefghijklmnopqrstuvwxyz0123456789_', min_size=1), min_size=1, max_size=3))
@settings(max_examples=200)
def test_parse_list_exact_match_with_hash(items):
    items_with_hash = [item + '#' + str(i) for i, item in enumerate(items)]
    formatted = ' '.join(items_with_hash)
    result = parse_list(formatted)

    assert result == items_with_hash, \
        f"parse_list should preserve items exactly, but {items_with_hash} -> {result}"
```

**Failing input**: `items=['0']` → `parse_list('0#0')` returns `['0#__Pyx_L1_']`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Build.Dependencies import parse_list

result = parse_list("lib#1")
assert result == ["lib#1"], f"Expected ['lib#1'], got {result}"
```

## Why This Is A Bug

The `parse_list` function is used to parse distutils directive values from Cython source files, such as library names and macro definitions. Library and macro names can legitimately contain `#` characters (e.g., versioned libraries like `lib#1`, or macros with `#` in the name). The function calls `strip_string_literals`, which treats `#` as a comment marker and replaces everything after it with a generated label. This causes silent corruption of valid input values.

## Fix

The issue is that `parse_list` calls `strip_string_literals` on line 128, which is designed to parse Python/Cython source code (where `#` starts comments), not simple value lists. Since `parse_list` is specifically for parsing distutils directive values (not full Python code), it should not strip comments.

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -125,7 +125,6 @@ def parse_list(s):
         s = s[1:-1]
         delimiter = ','
     else:
         delimiter = ' '
-    s, literals = strip_string_literals(s)
+    literals = {}
     def unquote(literal):
         literal = literal.strip()
@@ -133,5 +132,11 @@ def parse_list(s):
             return literals[literal[1:-1]]
         else:
             return literal
+
+    if delimiter == ',':
+        s_for_quotes, literals = strip_string_literals(s)
+        return [unquote(item) for item in s.split(delimiter) if item.strip()]
+
     return [unquote(item) for item in s.split(delimiter) if item.strip()]
```

A better fix would be to only strip quoted strings (not comments) when parsing the list:

```diff
--- a/Cython/Build/Dependencies.py
+++ b/Cython/Build/Dependencies.py
@@ -108,6 +108,25 @@ def update_pythran_extension(ext):
         except ValueError:
             pass

+def strip_quoted_strings(s):
+    """Strip only quoted strings from input, not comments."""
+    import re
+    literals = {}
+    counter = 0
+
+    def replace_quoted(m):
+        nonlocal counter
+        counter += 1
+        label = f"__Pyx_L{counter}_"
+        literals[label] = m.group(0)
+        return label
+
+    pattern = r"'(?:[^'\\\\]|\\\\.)*'|\"(?:[^\"\\\\]|\\\\.)*\""
+    new_s = re.sub(pattern, replace_quoted, s)
+    return new_s, literals

 def parse_list(s):
     """
@@ -125,7 +144,7 @@ def parse_list(s):
         s = s[1:-1]
         delimiter = ','
     else:
         delimiter = ' '
-    s, literals = strip_string_literals(s)
+    s, literals = strip_quoted_strings(s)
     def unquote(literal):
         literal = literal.strip()
```