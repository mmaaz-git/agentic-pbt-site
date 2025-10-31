# Bug Report: Cython.Tempita SyntaxError Missing Position Information

**Target**: `Cython.Tempita.Template._eval`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

SyntaxError exceptions raised from template expressions in Cython.Tempita lack line and column position information, while all other exception types (NameError, ValueError, TypeError) correctly include this information, making debugging template syntax errors unnecessarily difficult.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, assume, strategies as st
from Cython.Tempita import Template

@given(st.text(alphabet='0123456789+-*/', min_size=1, max_size=10))
def test_syntaxerror_includes_position(expr):
    assume('{{' not in expr and '}}' not in expr)

    content = f"Line 1\nLine 2\n{{{{{expr}}}}}"
    template = Template(content)

    try:
        template.substitute({})
    except SyntaxError as e:
        error_msg = str(e)
        assert 'line' in error_msg.lower() and 'column' in error_msg.lower(), \
            f"SyntaxError message lacks position info: '{error_msg}'"

if __name__ == "__main__":
    # Run the test
    test_syntaxerror_includes_position()
```

<details>

<summary>
**Failing input**: `/` (and other invalid Python expressions like `**`, `+++`, `+*`)
</summary>
```
  + Exception Group Traceback (most recent call last):
  |   File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 23, in <module>
  |     test_syntaxerror_includes_position()
  |     ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  |   File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 8, in test_syntaxerror_includes_position
  |     def test_syntaxerror_includes_position(expr):
  |                    ^^^
  |   File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
  |     raise the_error_hypothesis_found
  | ExceptionGroup: Hypothesis found 2 distinct failures. (2 sub-exceptions)
  +-+---------------- 1 ----------------
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 15, in test_syntaxerror_includes_position
    |     template.substitute({})
    |     ~~~~~~~~~~~~~~~~~~~^^^^
    |   File "Cython/Tempita/_tempita.py", line 186, in Cython.Tempita._tempita.Template.substitute
    |   File "Cython/Tempita/_tempita.py", line 197, in Cython.Tempita._tempita.Template._interpret
    |   File "Cython/Tempita/_tempita.py", line 225, in Cython.Tempita._tempita.Template._interpret_codes
    |   File "Cython/Tempita/_tempita.py", line 245, in Cython.Tempita._tempita.Template._interpret_code
    |   File "Cython/Tempita/_tempita.py", line 318, in Cython.Tempita._tempita.Template._eval
    |   File "Cython/Tempita/_tempita.py", line 307, in Cython.Tempita._tempita.Template._eval
    |   File "<string>", line 1, in <module>
    | ZeroDivisionError: division by zero at line 3 column 3
    | Falsifying example: test_syntaxerror_includes_position(
    |     expr='0/0',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/22/hypo.py:16
    +---------------- 2 ----------------
    | Traceback (most recent call last):
    |   File "Cython/Tempita/_tempita.py", line 307, in Cython.Tempita._tempita.Template._eval
    |   File "<string>", line 1
    |     /
    |     ^
    | SyntaxError: invalid syntax
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 15, in test_syntaxerror_includes_position
    |     template.substitute({})
    |     ~~~~~~~~~~~~~~~~~~~^^^^
    |   File "Cython/Tempita/_tempita.py", line 186, in Cython.Tempita._tempita.Template.substitute
    |   File "Cython/Tempita/_tempita.py", line 197, in Cython.Tempita._tempita.Template._interpret
    |   File "Cython/Tempita/_tempita.py", line 225, in Cython.Tempita._tempita.Template._interpret_codes
    |   File "Cython/Tempita/_tempita.py", line 245, in Cython.Tempita._tempita.Template._interpret_code
    |   File "Cython/Tempita/_tempita.py", line 318, in Cython.Tempita._tempita.Template._eval
    |   File "Cython/Tempita/_tempita.py", line 309, in Cython.Tempita._tempita.Template._eval
    | SyntaxError: invalid syntax in expression: /
    |
    | During handling of the above exception, another exception occurred:
    |
    | Traceback (most recent call last):
    |   File "/home/npc/pbt/agentic-pbt/worker_/22/hypo.py", line 18, in test_syntaxerror_includes_position
    |     assert 'line' in error_msg.lower() and 'column' in error_msg.lower(), \
    |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    | AssertionError: SyntaxError message lacks position info: 'invalid syntax in expression: /'
    | Falsifying example: test_syntaxerror_includes_position(
    |     expr='/',
    | )
    | Explanation:
    |     These lines were always and only run by failing examples:
    |         /home/npc/pbt/agentic-pbt/worker_/22/hypo.py:16
    +------------------------------------
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test SyntaxError - missing position info
content = "Line 1\nLine 2\n{{/}}"
template = Template(content)

print("Testing SyntaxError with template: {{/}}")
print("-" * 50)
try:
    template.substitute({})
except SyntaxError as e:
    print(f"SyntaxError: {e}")
print()

# Test NameError - has position info
content2 = "Line 1\nLine 2\n{{undefined}}"
template2 = Template(content2)

print("Testing NameError with template: {{undefined}}")
print("-" * 50)
try:
    template2.substitute({})
except NameError as e:
    print(f"NameError: {e}")
print()

# Test ValueError - has position info
content3 = "Line 1\nLine 2\n{{int('abc')}}"
template3 = Template(content3)

print("Testing ValueError with template: {{int('abc')}}")
print("-" * 50)
try:
    template3.substitute({})
except ValueError as e:
    print(f"ValueError: {e}")
print()

# Test TypeError - has position info
content4 = "Line 1\nLine 2\n{{len(123)}}"
template4 = Template(content4)

print("Testing TypeError with template: {{len(123)}}")
print("-" * 50)
try:
    template4.substitute({})
except TypeError as e:
    print(f"TypeError: {e}")
```

<details>

<summary>
SyntaxError lacks position info while other exceptions include it
</summary>
```
Testing SyntaxError with template: {{/}}
--------------------------------------------------
SyntaxError: invalid syntax in expression: /

Testing NameError with template: {{undefined}}
--------------------------------------------------
NameError: name 'undefined' is not defined at line 3 column 3

Testing ValueError with template: {{int('abc')}}
--------------------------------------------------
ValueError: invalid literal for int() with base 10: 'abc' at line 3 column 3

Testing TypeError with template: {{len(123)}}
--------------------------------------------------
TypeError: object of type 'int' has no len() at line 3 column 3
```
</details>

## Why This Is A Bug

This violates the expected contract that all template evaluation errors should include position information for debugging. The `_eval` method in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py:303-318` contains a nested try-except structure where:

1. **Lines 306-310**: SyntaxError is caught in an inner try-except block and a new SyntaxError is raised with only `'invalid syntax in expression: %s' % code`
2. **Lines 312-318**: The outer exception handler that adds position information via `_add_line_info(arg0, pos)` is never reached for SyntaxError
3. **All other exceptions** (NameError, ValueError, TypeError, etc.) bypass the inner catch and go through the outer handler, getting position info added correctly

This contradicts the module's design where:
- The `_add_line_info` method exists specifically to add `"at line X column Y"` to ALL error messages
- The TemplateError class documentation shows all errors should include position information
- No documentation indicates SyntaxError should be treated differently
- Users expect consistent error formatting across all exception types for effective debugging

## Relevant Context

The `_add_line_info` method at lines 373-378 formats position information as:
```python
def _add_line_info(self, msg, pos):
    msg = "%s at line %s column %s" % (msg, pos[0], pos[1])
    if self.name:
        msg += " in file %s" % self.name
    return msg
```

This method is designed to be applied to all exceptions but the nested try-except inadvertently prevents it from being applied to SyntaxError. The bug makes debugging template syntax errors more difficult than necessary since users don't immediately know where in their template the syntax error occurred.

Documentation examples show errors with position info (e.g., "NameError: name 'name' is not defined at line 1 column 6 in file tmpl") with no indication that SyntaxError should be an exception to this pattern.

## Proposed Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -303,14 +303,11 @@ class Template:
     def _eval(self, code, ns, pos):
         __traceback_hide__ = True
         try:
-            try:
-                value = eval(code, self.default_namespace, ns)
-            except SyntaxError as e:
-                raise SyntaxError(
-                    'invalid syntax in expression: %s' % code)
+            value = eval(code, self.default_namespace, ns)
             return value
+        except SyntaxError as e:
+            raise SyntaxError(
+                self._add_line_info('invalid syntax in expression: %s' % code, pos))
         except Exception as e:
             if getattr(e, 'args', None):
                 arg0 = e.args[0]
```