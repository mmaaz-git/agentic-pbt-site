# Bug Report: Cython.Tempita Parser Accepts Invalid Conditional Syntax

**Target**: `Cython.Tempita._tempita.parse_cond`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The template parser accepts invalid conditional block syntax including duplicate `else` clauses and `elif` after `else`, which should be syntax errors per Python syntax rules. Invalid clauses are silently ignored during execution.

## Property-Based Test

```python
@given(st.integers(min_value=1, max_value=5))
def test_template_rejects_duplicate_else(num_else_clauses):
    assume(num_else_clauses >= 2)

    else_blocks = ''.join([f"{{{{else}}}}{i}\n" for i in range(num_else_clauses)])
    content = f"{{{{if False}}}}\nA\n{else_blocks}{{{{endif}}}}"

    with pytest.raises(TemplateError):
        Template(content)
```

**Failing input**: Templates with duplicate `else` or `elif` after `else`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content1 = "{{if x}}A{{else}}B{{else}}C{{endif}}"
template1 = Template(content1)
print(template1.substitute({'x': False}))

content2 = "{{if x}}A{{else}}B{{elif y}}C{{endif}}"
template2 = Template(content2)
print(template2.substitute({'x': False, 'y': True}))
```

Both templates parse without error, even though they contain invalid syntax.

## Why This Is A Bug

Python (and most languages) enforce that:
1. Only one `else` clause is allowed per if block
2. `elif` cannot appear after `else`

The parser should reject these as syntax errors. The comment on line 291 (`# @@: if/else/else gets through`) indicates the author knew about this issue.

`parse_cond()` doesn't track whether an `else` has been seen, allowing multiple `else` clauses and `elif` clauses after `else` to be accepted. During execution, only the first matching clause runs, but the invalid syntax should have been rejected during parsing.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -800,6 +800,7 @@ def parse_cond(tokens, name, context):
 def parse_cond(tokens, name, context):
     start = tokens[0][1]
     pieces = []
+    seen_else = False
     context = context + ('if',)
     while 1:
         if not tokens:
@@ -809,6 +810,15 @@ def parse_cond(tokens, name, context):
         if (isinstance(tokens[0], tuple)
                 and tokens[0][0] == 'endif'):
             return ('cond', start) + tuple(pieces), tokens[1:]
+        if isinstance(tokens[0], tuple):
+            clause = tokens[0][0]
+            if clause == 'else':
+                if seen_else:
+                    raise TemplateError('Duplicate else clause', position=tokens[0][1], name=name)
+                seen_else = True
+            elif clause.startswith('elif ') and seen_else:
+                raise TemplateError('elif after else', position=tokens[0][1], name=name)
         next_chunk, tokens = parse_one_cond(tokens, name, context)
         pieces.append(next_chunk)
```