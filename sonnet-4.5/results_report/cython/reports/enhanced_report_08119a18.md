# Bug Report: Cython.Tempita sub() __name Parameter Leaks into Template Namespace

**Target**: `Cython.Tempita._tempita.sub`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `sub()` function's `__name` parameter is not removed from the kwargs dict before passing it to `substitute()`, causing it to leak into the template namespace where it can be accessed as a variable via `{{__name}}`.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for Cython.Tempita sub() __name parameter leak"""

import sys
import string
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, settings, strategies as st
from Cython.Tempita import sub

@given(st.text(alphabet=string.ascii_letters, min_size=1, max_size=20))
@settings(max_examples=100)
def test_sub_name_parameter_isolation(template_name):
    """Test that __name parameter does not leak into template namespace"""
    content = "{{__name}}"

    result = sub(content, __name=template_name)

    assert result == '', f"__name should not be accessible in template namespace, but got: {repr(result)}"

if __name__ == "__main__":
    test_sub_name_parameter_isolation()
```

<details>

<summary>
**Failing input**: `'A'` (or any other generated value)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 22, in <module>
    test_sub_name_parameter_isolation()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 12, in test_sub_name_parameter_isolation
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/34/hypo.py", line 19, in test_sub_name_parameter_isolation
    assert result == '', f"__name should not be accessible in template namespace, but got: {repr(result)}"
           ^^^^^^^^^^^^
AssertionError: __name should not be accessible in template namespace, but got: 'A'
Falsifying example: test_sub_name_parameter_isolation(
    template_name='A',  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction demonstrating __name parameter leak in Cython.Tempita.sub()"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import sub

# Test 1: Basic leak demonstration
content = "Template name: {{__name}}"
result = sub(content, __name='mytemplate.html')

print("Test 1: Basic __name leak")
print("-" * 40)
print(f"Template: {repr(content)}")
print(f"Call: sub(content, __name='mytemplate.html')")
print(f"Expected: 'Template name: ' (empty)")
print(f"Actual: {repr(result)}")
print()

# Test 2: __name alongside regular variables
content2 = "Name is: {{__name}}, foo is: {{foo}}"
result2 = sub(content2, __name='template.html', foo='bar')

print("Test 2: __name with other variables")
print("-" * 40)
print(f"Template: {repr(content2)}")
print(f"Call: sub(content2, __name='template.html', foo='bar')")
print(f"Expected: 'Name is: , foo is: bar'")
print(f"Actual: {repr(result2)}")
print()

# Test 3: Empty __name
content3 = "{{__name}}"
result3 = sub(content3, __name='')

print("Test 3: Empty __name value")
print("-" * 40)
print(f"Template: {repr(content3)}")
print(f"Call: sub(content3, __name='')")
print(f"Expected: '' (empty)")
print(f"Actual: {repr(result3)}")
print()

print("BUG CONFIRMED: __name parameter is accessible as template variable")
print("This violates the documented purpose of __name as a meta-parameter for error reporting")
```

<details>

<summary>
Output demonstrating __name leaking into template namespace
</summary>
```
Test 1: Basic __name leak
----------------------------------------
Template: 'Template name: {{__name}}'
Call: sub(content, __name='mytemplate.html')
Expected: 'Template name: ' (empty)
Actual: 'Template name: mytemplate.html'

Test 2: __name with other variables
----------------------------------------
Template: 'Name is: {{__name}}, foo is: {{foo}}'
Call: sub(content2, __name='template.html', foo='bar')
Expected: 'Name is: , foo is: bar'
Actual: 'Name is: template.html, foo is: bar'

Test 3: Empty __name value
----------------------------------------
Template: '{{__name}}'
Call: sub(content3, __name='')
Expected: '' (empty)
Actual: ''

BUG CONFIRMED: __name parameter is accessible as template variable
This violates the documented purpose of __name as a meta-parameter for error reporting
```
</details>

## Why This Is A Bug

The `__name` parameter is documented and intended as a meta-parameter for naming templates for error reporting purposes, not as a template variable. However, due to inconsistent parameter handling in the `sub()` function, it becomes accessible within templates as `{{__name}}`.

The bug occurs because of inconsistent parameter extraction at line 382 of `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py`:

- Line 382 uses `kw.get('__name')` which reads but doesn't remove the parameter from kwargs
- Line 383 correctly uses `kw.pop('delimeters')` to remove the legacy `delimeters` parameter
- Line 385 passes the unmodified `kw` dict (still containing `__name`) to `substitute()`

This violates several expectations:
1. **Python convention**: Parameters prefixed with double underscore (`__`) typically indicate special/internal attributes that shouldn't be directly accessible
2. **Documentation intent**: The `__name` parameter is documented as serving a meta-purpose (template naming for error messages), not as a template variable
3. **Code consistency**: The handling differs from the legacy `delimeters` parameter which is properly removed using `pop()`
4. **Principle of least surprise**: Users would not expect `{{__name}}` in their templates to expand to the template's name parameter

## Relevant Context

The `sub()` function is a convenience function in Cython's Tempita templating system that combines template creation and substitution in a single call. It accepts a `__name` parameter to set the template name for improved error messages when template parsing or execution fails.

The Template class constructor (starting at line 109) accepts a `name` parameter which is stored as `self.name` and used exclusively for error reporting (see line 377 where it's appended to error messages).

Source code location: [Cython/Tempita/_tempita.py:381-385](https://github.com/cython/cython/blob/master/Cython/Tempita/_tempita.py#L381-L385)

The inconsistency is clear when comparing how special parameters are handled:
- `delimeters` (legacy spelling): Properly removed with `pop()`
- `__name`: Incorrectly left in kwargs with `get()`

## Proposed Fix

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