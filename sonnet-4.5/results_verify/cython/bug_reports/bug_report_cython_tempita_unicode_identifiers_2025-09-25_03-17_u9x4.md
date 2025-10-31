# Bug Report: Cython.Tempita Unicode Identifier Rejection

**Target**: `Cython.Tempita._tempita.parse_default`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

Tempita's `{{default}}` statement rejects valid Python 3 Unicode identifiers using an ASCII-only regex that doesn't match Python 3's identifier rules.

## Property-Based Test

```python
@given(st.text(min_size=1).filter(str.isidentifier))
def test_default_statement_sets_variable_if_not_in_namespace(var_name):
    content = f"""
{{{{default {var_name} = 42}}}}
{{{{{var_name}}}}}
"""
    template = Template(content)
    result = template.substitute({})
    assert '42' in result
```

**Failing input**: `var_name='Aª'` (or 'café', 'π', etc. - any Unicode identifier)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

var_name = 'café'

content = f"""
{{{{default {var_name} = 42}}}}
{{{{{var_name}}}}}
"""

template = Template(content)
result = template.substitute({})
print(result)
```

**Output**: `TemplateError: Not a valid variable name for {{default}}: 'café' at line 2 column 3`

## Why This Is A Bug

Python 3 supports Unicode identifiers per PEP 3131. Variable names like 'café', 'π', and 'naïve' are valid Python identifiers and work with `eval()` and `exec()`. However, Tempita uses an ASCII-only regex for validating variable names in `{{default}}` statements.

Line 44 defines:
```python
var_re = re.compile(r'^[a-z_][a-z0-9_]*$', re.I)
```

This regex only matches ASCII letters [a-zA-Z], digits [0-9], and underscores. Line 892 in `parse_default()` uses this to validate variable names:

```python
if not var_re.search(var):
    raise TemplateError(
        "Not a valid variable name for {{default}}: %r"
        % var, position=pos, name=name)
```

Since Tempita already uses Python's `eval()` for expression evaluation, which supports Unicode identifiers, there's no technical reason to restrict identifier syntax beyond what Python allows.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -41,7 +41,6 @@ __all__ = ['TemplateError', 'Template', 'sub', 'bunch']

 in_re = re.compile(r'\s+in\s+')
-var_re = re.compile(r'^[a-z_][a-z0-9_]*$', re.I)

 def coerce_text(v):
     if not isinstance(v, str):
@@ -889,7 +888,7 @@ def parse_default(tokens, name, context):
         raise TemplateError(
             "{{default x, y = ...}} is not supported",
             position=pos, name=name)
-    if not var_re.search(var):
+    if not var.isidentifier():
         raise TemplateError(
             "Not a valid variable name for {{default}}: %r"
             % var, position=pos, name=name)
```