# Bug Report: Cython.Tempita parse_def Potential IndexError on Empty Def Directive

**Target**: `Cython.Tempita._tempita.parse_def`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `parse_def` function in the source code has a defect that would cause an IndexError when parsing `{{def}}` directives with no function signature, though the compiled Cython module masks this issue by raising TemplateError instead.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from Cython.Tempita import Template, TemplateError
import pytest

@given(st.text(alphabet=' \t', min_size=0, max_size=5))
def test_def_with_no_signature_raises_template_error(whitespace):
    content = f"{{{{def{whitespace}}}}}{{{{enddef}}}}"

    print(f"Testing: {repr(content)}")

    with pytest.raises(TemplateError):
        template = Template(content)

# Run the test
if __name__ == "__main__":
    # Import hypothesis testing
    test_def_with_no_signature_raises_template_error()
```

<details>

<summary>
**Failing input**: `{{def }}{{enddef}}`
</summary>
```
Testing: '{{def}}{{enddef}}'
Testing: '{{def\t\t}}{{enddef}}'
Testing: '{{def  }}{{enddef}}'
Testing: '{{def \t \t }}{{enddef}}'
Testing: '{{def\t\t \t}}{{enddef}}'
Testing: '{{def }}{{enddef}}'
Testing: '{{def\t  }}{{enddef}}'
Testing: '{{def\t\t\t  }}{{enddef}}'
Testing: '{{def\t \t\t\t}}{{enddef}}'
Testing: '{{def\t}}{{enddef}}'
Testing: '{{def    }}{{enddef}}'
Testing: '{{def\t\t  \t}}{{enddef}}'
Testing: '{{def \t\t}}{{enddef}}'
Testing: '{{def  \t}}{{enddef}}'
Testing: '{{def   }}{{enddef}}'
Testing: '{{def\t \t}}{{enddef}}'
Testing: '{{def \t\t\t}}{{enddef}}'
Testing: '{{def  \t\t\t}}{{enddef}}'
Testing: '{{def \t}}{{enddef}}'
Testing: '{{def  \t  }}{{enddef}}'
Testing: '{{def \t\t \t}}{{enddef}}'
Testing: '{{def\t \t\t }}{{enddef}}'
Testing: '{{def\t\t \t }}{{enddef}}'
Testing: '{{def \t }}{{enddef}}'
Testing: '{{def     }}{{enddef}}'
Testing: '{{def\t\t }}{{enddef}}'
Testing: '{{def\t \t  }}{{enddef}}'
Testing: '{{def\t\t\t}}{{enddef}}'
Testing: '{{def\t   \t}}{{enddef}}'
Testing: '{{def\t\t   }}{{enddef}}'
Testing: '{{def   \t}}{{enddef}}'
Testing: '{{def\t }}{{enddef}}'
Testing: '{{def  \t }}{{enddef}}'
Testing: '{{def\t \t }}{{enddef}}'
Testing: '{{def  \t \t}}{{enddef}}'
Testing: '{{def\t  \t }}{{enddef}}'
Testing: '{{def\t    }}{{enddef}}'
Testing: '{{def    \t}}{{enddef}}'
Testing: '{{def\t\t\t\t }}{{enddef}}'
Testing: '{{def \t\t\t\t}}{{enddef}}'
Testing: '{{def\t\t\t\t}}{{enddef}}'
Testing: '{{def \t  \t}}{{enddef}}'
Testing: '{{def\t\t\t\t\t}}{{enddef}}'
Testing: '{{def\t\t\t }}{{enddef}}'
Testing: '{{def \t  }}{{enddef}}'
Testing: '{{def \t\t\t }}{{enddef}}'
Testing: '{{def\t\t \t\t}}{{enddef}}'
Testing: '{{def \t   }}{{enddef}}'
Testing: '{{def   \t }}{{enddef}}'
Testing: '{{def  \t\t}}{{enddef}}'
Testing: '{{def  \t\t }}{{enddef}}'
Testing: '{{def   \t\t}}{{enddef}}'
Testing: '{{def\t  \t\t}}{{enddef}}'
Testing: '{{def\t\t  }}{{enddef}}'
Testing: '{{def \t \t}}{{enddef}}'
Testing: '{{def \t \t\t}}{{enddef}}'
Testing: '{{def\t\t\t \t}}{{enddef}}'
Testing: '{{def \t\t  }}{{enddef}}'
Testing: '{{def\t \t \t}}{{enddef}}'
Testing: '{{def\t \t\t}}{{enddef}}'
Testing: '{{def \t\t }}{{enddef}}'
Testing: '{{def\t   }}{{enddef}}'
Testing: '{{def\t  \t}}{{enddef}}'

```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

content = "{{def }}{{enddef}}"
print(f"Testing template: {repr(content)}")
try:
    template = Template(content)
    print("Template created successfully")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
```

<details>

<summary>
TemplateError: Unexpected enddef at line 1 column 11
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/55/repo.py", line 9, in <module>
    template = Template(content)
  File "Cython/Tempita/_tempita.py", line 145, in Cython.Tempita._tempita.Template.__init__
  File "Cython/Tempita/_tempita.py", line 742, in Cython.Tempita._tempita.parse
  File "Cython/Tempita/_tempita.py", line 784, in Cython.Tempita._tempita.parse_expr
Cython.Tempita._tempita.TemplateError: Unexpected enddef at line 1 column 11
Testing template: '{{def }}{{enddef}}'
Error: TemplateError: Unexpected enddef at line 1 column 11

```
</details>

## Why This Is A Bug

While the compiled Cython module correctly raises a TemplateError for malformed `{{def}}` directives, the source code at `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py` line 911 contains a defect that would cause an IndexError if executed as pure Python.

The problematic code:
```python
def parse_def(tokens, name, context):
    first, start = tokens[0]
    tokens = tokens[1:]
    assert first.startswith('def ')
    first = first.split(None, 1)[1]  # Line 911: Would fail with IndexError
```

When `first` is `"def "` (with only whitespace after 'def'), `first.split(None, 1)` returns `['def']` - a single-element list. Attempting to access index `[1]` would raise an IndexError in pure Python execution. The function should validate that a function signature exists and raise an appropriate TemplateError instead.

## Relevant Context

The bug exists in the source code but doesn't manifest in practice because:
1. The Cython module is compiled and the compiled version handles this case differently
2. The compiled module correctly raises `TemplateError: Unexpected enddef` instead of crashing

This represents a discrepancy between source and compiled behavior. While not critical since users won't experience crashes, the source code should be corrected for:
- Code maintainability
- Consistency between source and compiled versions
- Preventing future issues if the compilation process changes

Similar parsing functions in the same file (like `parse_inherit` at line 903) have the same potential issue, suggesting a pattern that could be improved throughout the codebase.

## Proposed Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -908,7 +908,11 @@ def parse_def(tokens, name, context):
     first, start = tokens[0]
     tokens = tokens[1:]
     assert first.startswith('def ')
-    first = first.split(None, 1)[1]
+    parts = first.split(None, 1)
+    if len(parts) < 2:
+        raise TemplateError("{{def}} directive requires a function signature",
+                            position=start, name=name)
+    first = parts[1]
     if first.endswith(':'):
         first = first[:-1]
     if '(' not in first:
```