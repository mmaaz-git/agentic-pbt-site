# Bug Report: Cython.Tempita parse_signature Last Argument Missing

**Target**: `Cython.Tempita._tempita.parse_signature`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_signature` function fails to include the last argument in function signatures, making template functions with any arguments completely unusable unless all arguments have default values.

## Property-Based Test

```python
@given(st.lists(st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
                min_size=1, max_size=5))
def test_parse_signature_preserves_all_arguments(self, arg_names):
    assume(len(set(arg_names)) == len(arg_names))

    sig_text = ', '.join(arg_names)
    sig_args, var_arg, var_kw, defaults = parse_signature(sig_text, name='test', pos=(1, 1))

    assert len(sig_args) == len(arg_names)
    assert sig_args == arg_names
```

**Failing input**: Any function signature, e.g., `"x"` returns `([], None, None, {})` instead of `(['x'], None, None, {})`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

template_content = """
{{def greet(name)}}
Hello, {{name}}!
{{enddef}}

{{greet('World')}}
"""

template = Template(template_content)
result = template.substitute({})
```

Output:
```
TypeError: Extra position arguments: 'World' at line 6 column 3
```

The function signature `greet(name)` is parsed as having zero arguments, so calling `greet('World')` fails because it thinks 'World' is an extra argument.

## Why This Is A Bug

In `parse_signature` (lines 938-1011), the code checks for `tokenize.ENDMARKER` or comma on line 967 to determine when to add an argument to `sig_args`. However, Python's tokenizer emits a `NEWLINE` token before `ENDMARKER` for single-line strings.

For signature `"x"`, the tokens are:
1. `NAME 'x'`
2. `NEWLINE ''`
3. `ENDMARKER ''`

After getting the NAME token 'x', the code calls `get_token()` (line 966) and receives `NEWLINE`. Since `NEWLINE` is neither `ENDMARKER` nor a comma, the condition on line 967 is false. The code then falls through without adding the argument to `sig_args`, eventually returning with an incomplete signature.

This affects:
- Signature `"x"` → returns `[]` instead of `['x']`
- Signature `"x, y"` → returns `['x']` instead of `['x', 'y']`
- Signature `"x, y, z"` → returns `['x', 'y']` instead of `['x', 'y', 'z']`

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -964,7 +964,7 @@ def parse_signature(sig_text, name, pos):
                             position=pos, name=name)
         var_name = tok_string
         tok_type, tok_string = get_token()
-        if tok_type == tokenize.ENDMARKER or (tok_type == tokenize.OP and tok_string == ','):
+        if tok_type in (tokenize.ENDMARKER, tokenize.NEWLINE) or (tok_type == tokenize.OP and tok_string == ','):
             if var_arg_type == '*':
                 var_arg = var_name
             elif var_arg_type == '**':
```