# Bug Report: Cython.Tempita Multiple Else Clauses Accepted

**Target**: `Cython.Tempita._tempita.parse_one_cond`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Template parser incorrectly accepts if statements with multiple else clauses and elif clauses after else, violating standard conditional statement syntax.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Tempita import Template, TemplateError

@given(st.booleans())
def test_multiple_else_clauses_rejected(condition):
    content = """
{{if x}}
a
{{else}}
b
{{else}}
c
{{endif}}
"""

    template = Template(content)
    with pytest.raises(TemplateError):
        template.substitute({'x': condition})
```

**Failing input**: Any template with duplicate else clauses

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = """
{{if x}}
true_branch
{{else}}
first_else
{{else}}
second_else
{{endif}}
"""

template = Template(content)
result = template.substitute({'x': False})
print(result)
```

Output: Template is accepted and renders "first_else", silently ignoring the second else clause.

## Why This Is A Bug

The code has a TODO comment on line 291: "# @@: if/else/else gets through", explicitly acknowledging this is incorrect behavior that should be fixed. Standard conditional syntax requires:
- At most one else clause per if statement
- No elif clauses after an else clause

The parser should reject these malformed templates during parsing, but instead silently accepts them and uses only the first else clause, potentially hiding template syntax errors from users.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -800,6 +800,7 @@ def parse_cond(tokens, name, context):
     start = tokens[0][1]
     pieces = []
     context = context + ('if',)
+    seen_else = False
     while 1:
         if not tokens:
             raise TemplateError(
@@ -808,11 +809,12 @@ def parse_cond(tokens, name, context):
         if (isinstance(tokens[0], tuple)
                 and tokens[0][0] == 'endif'):
             return ('cond', start) + tuple(pieces), tokens[1:]
-        next_chunk, tokens = parse_one_cond(tokens, name, context)
+        next_chunk, tokens, seen_else = parse_one_cond(tokens, name, context, seen_else)
         pieces.append(next_chunk)


-def parse_one_cond(tokens, name, context):
+def parse_one_cond(tokens, name, context, seen_else=False):
     (first, pos), tokens = tokens[0], tokens[1:]
     content = []
     if first.endswith(':'):
@@ -820,9 +822,17 @@ def parse_one_cond(tokens, name, context):
     if first.startswith('if '):
         part = ('if', pos, first[3:].lstrip(), content)
     elif first.startswith('elif '):
+        if seen_else:
+            raise TemplateError(
+                'elif after else',
+                position=pos, name=name)
         part = ('elif', pos, first[5:].lstrip(), content)
     elif first == 'else':
+        if seen_else:
+            raise TemplateError(
+                'Multiple else clauses',
+                position=pos, name=name)
         part = ('else', pos, None, content)
+        seen_else = True
     else:
         assert 0, "Unexpected token %r at %s" % (first, pos)
     while 1:
@@ -835,7 +845,7 @@ def parse_one_cond(tokens, name, context):
                  or tokens[0][0].startswith('elif ')
                  or tokens[0][0] == 'else')):
             return part, tokens
-        next_chunk, tokens = parse_expr(tokens, name, context)
-        content.append(next_chunk)
+    return part, tokens, seen_else
```