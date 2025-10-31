# Bug Report: Cython.Tempita parse_inherit IndexError on Empty Directive

**Target**: `Cython.Tempita._tempita.parse_inherit`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_inherit` function crashes with an IndexError instead of raising a proper TemplateError when parsing malformed `{{inherit}}` directives that have no expression after the keyword.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Tempita import Template, TemplateError

@given(st.text(alphabet=' \t', min_size=0, max_size=5))
def test_inherit_with_no_expression_raises_template_error(whitespace):
    content = f"{{{{inherit{whitespace}}}}}"

    with pytest.raises(TemplateError):
        template = Template(content)
```

**Failing input**: `{{inherit }}` or `{{inherit}}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = "{{inherit }}"
template = Template(content)
```

**Expected**: `TemplateError` with message about missing inherit expression

**Actual**: `IndexError: list index out of range`

## Why This Is A Bug

In `Cython/Tempita/_tempita.py` line 903:

```python
def parse_inherit(tokens, name, context):
    first, pos = tokens[0]
    assert first.startswith('inherit ')
    expr = first.split(None, 1)[1]  # Line 903: Crashes if no expression after 'inherit'
    return ('inherit', pos, expr), tokens[1:]
```

When `first` is `"inherit "` or `"inherit"`, the expression `first.split(None, 1)` returns `['inherit']` (one element). Accessing index `[1]` raises IndexError.

The function should validate that an expression exists after the `inherit` keyword and raise a descriptive TemplateError instead.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -900,7 +900,10 @@ def parse_default(tokens, name, context):
 def parse_inherit(tokens, name, context):
     first, pos = tokens[0]
     assert first.startswith('inherit ')
-    expr = first.split(None, 1)[1]
+    parts = first.split(None, 1)
+    if len(parts) < 2:
+        raise TemplateError("{{inherit}} directive requires an expression", position=pos, name=name)
+    expr = parts[1]
     return ('inherit', pos, expr), tokens[1:]
```