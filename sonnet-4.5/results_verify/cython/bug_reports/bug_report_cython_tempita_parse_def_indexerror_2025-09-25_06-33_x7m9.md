# Bug Report: Cython.Tempita parse_def IndexError on Whitespace-Only Input

**Target**: `Cython.Tempita._tempita.parse_def`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_def` function crashes with IndexError instead of raising a meaningful TemplateError when parsing `{{def }}` (with only whitespace after the keyword).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Tempita import Template, TemplateError


@given(st.text(alphabet=' \t', min_size=0, max_size=5))
@settings(max_examples=100)
def test_parse_def_whitespace_only(whitespace):
    content = f"{{{{def{whitespace}}}}}{{{{enddef}}}}"

    try:
        template = Template(content)
        assert False, "Should raise TemplateError"
    except Exception as e:
        assert isinstance(e, TemplateError), f"Should be TemplateError, got {type(e).__name__}"
```

**Failing input**: Template with `{{def }}{{enddef}}` (whitespace only after keyword)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = "{{def }}{{enddef}}"

try:
    template = Template(content)
except IndexError as e:
    print(f"IndexError: {e}")
    print(f"Expected: TemplateError with message about missing function name")
    print(f"Actual: IndexError: list index out of range")
```

## Why This Is A Bug

Line 911 in `Cython/Tempita/_tempita.py` attempts to access index 1 of a split result without checking length:
```python
first = first.split(None, 1)[1]
```

When `first` is "def " (with only trailing whitespace), `first.split(None, 1)` returns `['def']`. Accessing `[1]` raises IndexError.

The function should raise a meaningful TemplateError indicating that the def directive requires a function name, but instead crashes with an unhelpful IndexError.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -908,7 +908,11 @@ def parse_def(tokens, name, context):
     first, start = tokens[0]
     tokens = tokens[1:]
     assert first.startswith('def ')
-    first = first.split(None, 1)[1]
+    parts = first.split(None, 1)
+    if len(parts) < 2:
+        raise TemplateError(
+            "{{def}} requires a function name",
+            position=start, name=name)
+    first = parts[1]
     if first.endswith(':'):
         first = first[:-1]
     if '(' not in first:
```