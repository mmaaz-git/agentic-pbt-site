# Bug Report: Cython.Tempita sub() __name Parameter Leaks into Template Namespace

**Target**: `Cython.Tempita._tempita.sub`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `sub()` function's `__name` parameter is not removed from the kwargs dict before passing it to `substitute()`, causing it to leak into the template namespace where it can be accessed as a variable.

## Property-Based Test

```python
@given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=20))
@settings(max_examples=100)
def test_sub_name_parameter_isolation(template_name):
    content = "{{__name}}"

    result = sub(content, __name=template_name)

    assert result == '', f"__name should not be accessible in template namespace"
```

**Failing input**: Any use of `sub()` with `__name` parameter and template that references `{{__name}}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import sub

content = "Template name: {{__name}}"
result = sub(content, __name='mytemplate.html')

print(f"Result: {result}")
print(f"Expected: 'Template name: ' (empty)")
print(f"Actual: 'Template name: mytemplate.html'")
print()
print("Bug: __name parameter is accessible as template variable")
```

## Why This Is A Bug

Line 382 in `Cython/Tempita/_tempita.py` uses `kw.get('__name')` to extract the template name, but doesn't remove it from `kw`:

```python
name = kw.get('__name')
```

Line 383 correctly uses `kw.pop('delimeters')` to remove the legacy parameter:
```python
delimeters = kw.pop('delimeters') if 'delimeters' in kw else None
```

This inconsistency suggests `__name` should also be popped. Otherwise, line 385 passes the unmodified `kw` (still containing `__name`) to `substitute()`, making it available as a template variable.

The `__name` parameter is documented as a way to set the template name for error reporting, not as a template variable. Users would not expect `{{__name}}` to expand to the template name.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -379,7 +379,7 @@ class Template:


 def sub(content, delimiters=None, **kw):
-    name = kw.get('__name')
+    name = kw.pop('__name', None)
     delimeters = kw.pop('delimeters') if 'delimeters' in kw else None  # for legacy code
     tmpl = Template(content, name=name, delimiters=delimiters, delimeters=delimeters)
     return tmpl.substitute(kw)
```