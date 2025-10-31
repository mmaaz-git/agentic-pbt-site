# Bug Report: Cython.Debugger CyBreak Complete Duplicate Suggestions

**Target**: `Cython.Debugger.libcython.CyBreak.complete`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `CyBreak.complete` method fails to filter out already-typed function names when `word` is empty, due to `text[:-0]` evaluating to an empty string, causing duplicate suggestions.

## Property-Based Test

```python
from hypothesis import given, strategies as st


def complete_unqualified_logic(text, word, all_names):
    word = word or ""
    seen = set(text[:-len(word)].split())
    return [n for n in all_names if n.startswith(word) and n not in seen]


@given(st.text(min_size=1, alphabet=st.characters(whitelist_categories=('Lu', 'Ll'))))
def test_complete_with_empty_word(funcname):
    word = ""
    text = f"cy break {funcname} "
    all_names = [funcname, "other_func", "another_func"]

    result = complete_unqualified_logic(text, word, all_names)

    assert funcname in result
```

**Failing input**: `funcname = "spam"`, `word = ""`

## Reproducing the Bug

```python
text = "cy break spam "
word = ""

seen = set(text[:-len(word)].split())
print(f"text[:-0] = {repr(text[:-0])}")
print(f"seen = {seen}")

all_names = ["spam", "eggs", "ham"]
result = [n for n in all_names if n.startswith(word) and n not in seen]
print(f"result = {result}")
```

Output:
```
text[:-0] = ''
seen = set()
result = ['spam', 'eggs', 'ham']
```

The already-typed function name "spam" appears in the completion suggestions even though it's already in the command line.

## Why This Is A Bug

1. **Python slicing quirk**: In Python, `text[:-0]` evaluates to `""` (empty string), not `text`. This causes `seen` to be empty when `word=""`.

2. **User-facing issue**: When completing at a word boundary (e.g., after typing "cy break spam " and pressing TAB), the completion suggests "spam" again instead of filtering it out.

3. **Inconsistent UX**: Users expect completion to suggest new names, not repeat what they've already typed.

## Fix

```diff
--- a/Cython/Debugger/libcython.py
+++ b/Cython/Debugger/libcython.py
@@ -954,7 +954,10 @@ class CyBreak(CythonCommand):
         words = text.strip().split()
         if not words or '.' not in words[-1]:
             # complete unqualified
-            seen = set(text[:-len(word)].split())
+            if len(word) == 0:
+                seen = set(text.split())
+            else:
+                seen = set(text[:-len(word)].split())
             return [n for n in all_names
                           if n.startswith(word) and n not in seen]
```