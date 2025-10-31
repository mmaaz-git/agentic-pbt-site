# Bug Report: Cython.Tempita parse_default IndexError on Empty Directive

**Target**: `Cython.Tempita._tempita.parse_default`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_default` function crashes with an IndexError instead of raising a proper TemplateError when parsing malformed `{{default}}` directives that have no content after the keyword.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Tempita import Template, TemplateError

@given(st.text(alphabet=' \t', min_size=0, max_size=5))
def test_default_with_no_expression_raises_template_error(whitespace):
    content = f"{{{{default{whitespace}}}}}"

    with pytest.raises(TemplateError):
        template = Template(content)
```

**Failing input**: `{{default }}` or `{{default}}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = "{{default }}"
template = Template(content)
```

**Expected**: `TemplateError` with message about malformed default directive

**Actual**: `IndexError: list index out of range`

## Why This Is A Bug

In `Cython/Tempita/_tempita.py` line 881:

```python
def parse_default(tokens, name, context):
    first, pos = tokens[0]
    assert first.startswith('default ')
    first = first.split(None, 1)[1]  # Line 881: Crashes if no content after 'default'
```

When `first` is `"default "` or `"default"`, the expression `first.split(None, 1)` returns a list with only one element `['default']`. Accessing index `[1]` raises an IndexError.

The function should validate that there is content after the `default` keyword and raise a descriptive TemplateError instead of crashing with an IndexError.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -878,7 +878,10 @@ def parse_default(tokens, name, context):
 def parse_default(tokens, name, context):
     first, pos = tokens[0]
     assert first.startswith('default ')
-    first = first.split(None, 1)[1]
+    parts = first.split(None, 1)
+    if len(parts) < 2:
+        raise TemplateError("{{default}} directive requires a variable assignment", position=pos, name=name)
+    first = parts[1]
     parts = first.split('=', 1)
     if len(parts) == 1:
         raise TemplateError(
```