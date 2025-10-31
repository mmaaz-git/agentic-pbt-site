# Bug Report: Cython.Tempita parse_def IndexError on Empty Directive

**Target**: `Cython.Tempita._tempita.parse_def`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `parse_def` function crashes with an IndexError instead of raising a proper TemplateError when parsing malformed `{{def}}` directives that have no function signature after the keyword.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Tempita import Template, TemplateError

@given(st.text(alphabet=' \t', min_size=0, max_size=5))
def test_def_with_no_signature_raises_template_error(whitespace):
    content = f"{{{{def{whitespace}}}}}{{{{enddef}}}}"

    with pytest.raises(TemplateError):
        template = Template(content)
```

**Failing input**: `{{def }}{{enddef}}` or `{{def}}{{enddef}}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = "{{def }}{{enddef}}"
template = Template(content)
```

**Expected**: `TemplateError` with message about malformed def directive

**Actual**: `IndexError: list index out of range`

## Why This Is A Bug

In `Cython/Tempita/_tempita.py` line 911:

```python
def parse_def(tokens, name, context):
    first, start = tokens[0]
    tokens = tokens[1:]
    assert first.startswith('def ')
    first = first.split(None, 1)[1]  # Line 911: Crashes if no signature after 'def'
    if first.endswith(':'):
        first = first[:-1]
    # ...
```

When `first` is `"def "` or `"def"`, the expression `first.split(None, 1)` returns `['def']` (one element). Accessing index `[1]` raises IndexError.

The function should validate that a function signature exists after the `def` keyword and raise a descriptive TemplateError instead.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -908,7 +908,10 @@ def parse_def(tokens, name, context):
     first, start = tokens[0]
     tokens = tokens[1:]
     assert first.startswith('def ')
-    first = first.split(None, 1)[1]
+    parts = first.split(None, 1)
+    if len(parts) < 2:
+        raise TemplateError("{{def}} directive requires a function signature", position=start, name=name)
+    first = parts[1]
     if first.endswith(':'):
         first = first[:-1]
     if '(' not in first:
```