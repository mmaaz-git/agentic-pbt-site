# Bug Report: Cython.Tempita parse_signature Missing Last Argument

**Target**: `Cython.Tempita._tempita.parse_signature`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_signature` function fails to parse the last function argument when it has no default value, causing template functions to be completely unusable for basic function calls with positional arguments.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from Cython.Tempita._tempita import parse_signature

@given(st.lists(st.text(alphabet='abcdefg', min_size=1, max_size=5).filter(str.isidentifier),
                min_size=1, max_size=3, unique=True))
def test_parse_signature_preserves_all_arguments(arg_names):
    sig_text = ', '.join(arg_names)
    sig_args, _, _, _ = parse_signature(sig_text, "test", (1, 1))

    assert len(sig_args) == len(arg_names), \
        f"Expected {len(arg_names)} args, got {len(sig_args)}"

    for name in arg_names:
        assert name in sig_args
```

**Failing input**: Any signature string, e.g., `"name"` yields `sig_args=[]` instead of `sig_args=['name']`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita._tempita import parse_signature
from Cython.Tempita import Template

result = parse_signature("name", "test", (1, 1))
sig_args, _, _, _ = result
print(f"Signature 'name' parsed as: {sig_args}")

result2 = parse_signature("name, greeting", "test", (1, 1))
sig_args2, _, _, _ = result2
print(f"Signature 'name, greeting' parsed as: {sig_args2}")

content = "{{def greet(name)}}Hello, {{name}}!{{enddef}}{{greet('World')}}"
template = Template(content)
result = template.substitute({})
```

**Expected output**:
```
Signature 'name' parsed as: ['name']
Signature 'name, greeting' parsed as: ['name', 'greeting']
Hello, World!
```

**Actual output**:
```
Signature 'name' parsed as: []
Signature 'name, greeting' parsed as: ['name']
TypeError: Extra position arguments: 'World'
```

## Why This Is A Bug

The tokenizer for signature parsing emits `NAME('name'), NEWLINE(''), ENDMARKER('')` but the code at line 967 only checks for `ENDMARKER` or comma (`,`), not `NEWLINE`. This causes the last argument to never be added to `sig_args`.

**Root cause**: Line 967 in `Cython/Tempita/_tempita.py`:
```python
if tok_type == tokenize.ENDMARKER or (tok_type == tokenize.OP and tok_string == ','):
```

This condition doesn't handle `NEWLINE` tokens. When parsing `"name"`, the token sequence is:
1. Get `NAME('name')` at line 956
2. Store `'name'` in `var_name` at line 965
3. Get `NEWLINE('')` at line 966
4. Check fails at line 967 (not ENDMARKER, not comma)
5. Loop continues to next iteration
6. Get `ENDMARKER` at line 956
7. Break at line 957 without processing `var_name`

For signatures with defaults like `"name='default'"`, it works because the `=` operator triggers special handling (lines 980-1000) that explicitly checks for ENDMARKER at line 995.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -964,7 +964,7 @@ def parse_signature(sig_text, name, pos):
                                 position=pos, name=name)
         var_name = tok_string
         tok_type, tok_string = get_token()
-        if tok_type == tokenize.ENDMARKER or (tok_type == tokenize.OP and tok_string == ','):
+        if tok_type in (tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL) or (tok_type == tokenize.OP and tok_string == ','):
             if var_arg_type == '*':
                 var_arg = var_name
             elif var_arg_type == '**':
@@ -991,7 +991,7 @@ def parse_signature(sig_text, name, pos):
                 raise TemplateError('Invalid signature: (%s)' % sig_text,
                                     position=pos, name=name)
                 if (not nest_count and
-                        (tok_type == tokenize.ENDMARKER or (tok_type == tokenize.OP and tok_string == ','))):
+                        (tok_type in (tokenize.ENDMARKER, tokenize.NEWLINE, tokenize.NL) or (tok_type == tokenize.OP and tok_string == ','))):
                     default_expr = isolate_expression(sig_text, start_pos, end_pos)
                     defaults[var_name] = default_expr
                     sig_args.append(var_name)
```

The fix adds `tokenize.NEWLINE` and `tokenize.NL` to the condition check, allowing the parser to correctly handle the newline token that appears before ENDMARKER.