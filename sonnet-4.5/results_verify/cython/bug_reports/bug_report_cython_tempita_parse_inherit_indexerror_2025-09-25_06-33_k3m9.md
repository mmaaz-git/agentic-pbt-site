# Bug Report: Cython.Tempita parse_inherit IndexError on Whitespace-Only Input

**Target**: `Cython.Tempita._tempita.parse_inherit`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_inherit` function crashes with IndexError instead of raising a meaningful TemplateError when parsing `{{inherit }}` (with only whitespace after the keyword).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Tempita import Template, TemplateError


@given(st.text(alphabet=' \t', min_size=0, max_size=5))
@settings(max_examples=100)
def test_parse_inherit_whitespace_only(whitespace):
    content = f"{{{{inherit{whitespace}}}}}"

    try:
        template = Template(content)
        assert False, "Should raise TemplateError"
    except Exception as e:
        assert isinstance(e, TemplateError), f"Should be TemplateError, got {type(e).__name__}"
```

**Failing input**: Template with `{{inherit }}` (whitespace only after keyword)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = "{{inherit }}"

try:
    template = Template(content)
except IndexError as e:
    print(f"IndexError: {e}")
    print(f"Expected: TemplateError with message about missing template name")
    print(f"Actual: IndexError: list index out of range")
```

## Why This Is A Bug

Line 903 in `Cython/Tempita/_tempita.py` attempts to access index 1 of a split result without checking length:
```python
expr = first.split(None, 1)[1]
```

When `first` is "inherit " (with only trailing whitespace), `first.split(None, 1)` returns `['inherit']`. Accessing `[1]` raises IndexError.

The function should raise a meaningful TemplateError indicating that the inherit directive requires a template name, but instead crashes with an unhelpful IndexError.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -900,7 +900,11 @@ def parse_inherit(tokens, name, context):
     first, pos = tokens[0]
     assert first.startswith('inherit ')
-    expr = first.split(None, 1)[1]
+    parts = first.split(None, 1)
+    if len(parts) < 2:
+        raise TemplateError(
+            "{{inherit}} requires a template name",
+            position=pos, name=name)
+    expr = parts[1]
     return ('inherit', pos, expr), tokens[1:]
```