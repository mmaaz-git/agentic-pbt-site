# Bug Report: Cython.Tempita Parser Accepts Duplicate else Clauses

**Target**: `Cython.Tempita._tempita.parse_cond`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The template parser accepts multiple `else` clauses in a single if/elif/else block, which should be a syntax error. Only the first `else` is executed, and subsequent ones are silently ignored.

## Property-Based Test

```python
@given(st.integers(), st.integers(), st.integers())
def test_template_rejects_duplicate_else(val1, val2, val3):
    content = f"""
{{{{if False}}}}
  {val1}
{{{{else}}}}
  {val2}
{{{{else}}}}
  {val3}
{{{{endif}}}}
"""
    with pytest.raises(TemplateError, match=r"duplicate.*else|multiple.*else"):
        template = Template(content)
```

**Failing input**: Any template with duplicate `else` clauses

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = """
{{if False}}
  A
{{else}}
  B
{{else}}
  C
{{endif}}
"""

template = Template(content)
result = template.substitute({})
print(result)
```

Output:
```
  B
```

The template is parsed without error, but only the first `else` clause is executed. The duplicate `else` is silently accepted.

## Why This Is A Bug

The parser should validate that each if/elif/else block has at most one `else` clause, similar to Python's syntax rules. The comment on line 291 (`# @@: if/else/else gets through`) indicates the author was aware of this issue.

Currently, `parse_cond()` keeps calling `parse_one_cond()` and appending results without checking if an `else` has already been seen. This allows invalid templates to be parsed and silently ignore duplicate clauses.

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
@@ -808,6 +809,11 @@ def parse_cond(tokens, name, context):
                 position=start, name=name)
         if (isinstance(tokens[0], tuple)
                 and tokens[0][0] == 'endif'):
+            return ('cond', start) + tuple(pieces), tokens[1:]
+        if (isinstance(tokens[0], tuple) and tokens[0][0] == 'else'):
+            if seen_else:
+                raise TemplateError('Duplicate else clause',
+                                    position=tokens[0][1], name=name)
+            seen_else = True
         next_chunk, tokens = parse_one_cond(tokens, name, context)
         pieces.append(next_chunk)
```