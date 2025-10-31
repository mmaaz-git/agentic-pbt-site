# Bug Report: Cython.Tempita Parser Silently Accepts Invalid Templates with Duplicate else Clauses

**Target**: `Cython.Tempita._tempita.parse_cond`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Cython.Tempita template parser incorrectly accepts templates containing multiple `else` clauses within a single if/elif/else block, silently ignoring all but the first `else` clause instead of raising a syntax error.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import pytest
from hypothesis import given
import hypothesis.strategies as st
from Cython.Tempita import Template, TemplateError

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

# Run the test
test_template_rejects_duplicate_else()
```

<details>

<summary>
**Failing input**: `val1=0, val2=0, val3=0` (fails for all inputs)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 24, in <module>
    test_template_rejects_duplicate_else()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 10, in test_template_rejects_duplicate_else
    def test_template_rejects_duplicate_else(val1, val2, val3):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 20, in test_template_rejects_duplicate_else
    with pytest.raises(TemplateError, match=r"duplicate.*else|multiple.*else"):
         ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/_pytest/raises.py", line 712, in __exit__
    fail(f"DID NOT RAISE {self.expected_exceptions[0]!r}")
    ~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/_pytest/outcomes.py", line 177, in fail
    raise Failed(msg=reason, pytrace=pytrace)
Failed: DID NOT RAISE <class 'Cython.Tempita._tempita.TemplateError'>
Falsifying example: test_template_rejects_duplicate_else(
    # The test always failed when commented parts were varied together.
    val1=0,  # or any other generated value
    val2=0,  # or any other generated value
    val3=0,  # or any other generated value
)
```
</details>

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

<details>

<summary>
Output shows only first else clause is executed
</summary>
```
  B

```
</details>

## Why This Is A Bug

This violates the expected behavior of template parsers and contradicts Python's syntax rules which Tempita is designed to mimic. The parser should reject templates with duplicate `else` clauses as syntactically invalid, similar to how Python raises a `SyntaxError` for duplicate else statements.

The bug allows syntactically invalid templates to be silently accepted, with the second and subsequent `else` clauses being parsed but never executed. This can lead to confusion and hidden bugs in templates where code appears to exist but is unreachable. The developer was aware of this issue, as evidenced by the comment on line 291 of `_tempita.py`: `# @@: if/else/else gets through`, indicating this was a known deficiency that should be fixed.

## Relevant Context

The issue occurs in the `parse_cond` function at `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py:800-813`. This function loops through tokens calling `parse_one_cond` repeatedly without tracking whether an `else` clause has already been encountered. When the template is executed, the `_interpret_if` method (lines 289-301) processes conditions sequentially and stops at the first truthy condition, treating any `else` as automatically true (line 296), which explains why only the first `else` executes.

Other popular template engines like Jinja2 and Django Templates correctly reject duplicate `else` clauses with syntax errors. The Tempita documentation shows examples with at most one `else` per if block, suggesting this should be the expected behavior.

## Proposed Fix

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
@@ -810,6 +811,12 @@ def parse_cond(tokens, name, context):
                 and tokens[0][0] == 'endif'):
             return ('cond', start) + tuple(pieces), tokens[1:]
         next_chunk, tokens = parse_one_cond(tokens, name, context)
+        if next_chunk[0] == 'else':
+            if seen_else:
+                raise TemplateError(
+                    'Duplicate else clause in conditional',
+                    position=next_chunk[1], name=name)
+            seen_else = True
         pieces.append(next_chunk)
```