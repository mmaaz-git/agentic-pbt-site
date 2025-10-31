# Bug Report: Cython.Tempita parse_default IndexError on Whitespace-Only Input

**Target**: `Cython.Tempita._tempita.parse_default`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_default` function crashes with IndexError instead of raising a meaningful TemplateError when parsing `{{default }}` (with only whitespace after the keyword).

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from Cython.Tempita import Template, TemplateError


@given(st.text(alphabet=' \t', min_size=0, max_size=5))
@settings(max_examples=100)
def test_parse_default_whitespace_only(whitespace):
    content = f"{{{{default{whitespace}}}}}"

    try:
        template = Template(content)
        assert False, "Should raise TemplateError"
    except Exception as e:
        assert isinstance(e, TemplateError), f"Should be TemplateError, got {type(e).__name__}"
        assert 'no = found' in str(e) or 'Not a valid variable name' in str(e)
```

**Failing input**: Template with `{{default }}` (whitespace only after keyword)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = "{{default }}"

try:
    template = Template(content)
except IndexError as e:
    print(f"IndexError: {e}")
    print(f"Expected: TemplateError with message about missing '='")
    print(f"Actual: IndexError: list index out of range")
except Exception as e:
    print(f"Got: {type(e).__name__}: {e}")
```

## Why This Is A Bug

Line 881 in `Cython/Tempita/_tempita.py` attempts to access index 1 of a split result:
```python
first = first.split(None, 1)[1]
```

When `first` is "default " (with only trailing whitespace), `first.split(None, 1)` returns `['default']` (a single-element list) because `split(None)` strips whitespace. Accessing `[1]` raises IndexError.

The function should raise a meaningful TemplateError indicating the syntax is invalid, but instead crashes with an unhelpful IndexError. This makes debugging template syntax errors more difficult.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -878,7 +878,11 @@ def parse_default(tokens, name, context):
     first, pos = tokens[0]
     assert first.startswith('default ')
-    first = first.split(None, 1)[1]
+    parts_split = first.split(None, 1)
+    if len(parts_split) < 2:
+        raise TemplateError(
+            "{{default}} requires a variable and value: {{default var=value}}",
+            position=pos, name=name)
+    first = parts_split[1]
     parts = first.split('=', 1)
     if len(parts) == 1:
         raise TemplateError(
```