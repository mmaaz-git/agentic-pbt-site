# Bug Report: Cython.Tempita Parser Accepts Invalid Conditional Syntax

**Target**: `Cython.Tempita._tempita.parse_cond`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The Cython.Tempita template parser accepts syntactically invalid conditional blocks containing duplicate `else` clauses and `elif` after `else`, silently ignoring the invalid clauses instead of raising a TemplateError during parsing.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import pytest
from hypothesis import given, assume, strategies as st
from Cython.Tempita import Template, TemplateError

@given(st.integers(min_value=1, max_value=5))
def test_template_rejects_duplicate_else(num_else_clauses):
    """Test that templates with duplicate else clauses are rejected."""
    assume(num_else_clauses >= 2)

    else_blocks = ''.join([f"{{{{else}}}}{i}\n" for i in range(num_else_clauses)])
    content = f"{{{{if False}}}}\nA\n{else_blocks}{{{{endif}}}}"

    print(f"\nTesting with {num_else_clauses} else clauses:")
    print(f"Template content: {repr(content)}")

    with pytest.raises(TemplateError):
        Template(content)

if __name__ == "__main__":
    # Run the test with a specific failing case
    print("Running property-based test for duplicate else clauses...")
    print("=" * 60)

    def run_test():
        for num_else in [2, 3, 4]:
            print(f"\nTesting with {num_else} else clauses:")
            else_blocks = ''.join([f"{{{{else}}}}{i}\n" for i in range(num_else)])
            content = f"{{{{if False}}}}\nA\n{else_blocks}{{{{endif}}}}"
            print(f"Template content: {repr(content)}")

            try:
                template = Template(content)
                print(f"ERROR: Template was accepted (should have raised TemplateError)")
                result = template.substitute({})
                print(f"Result when executed: {repr(result)}")
            except TemplateError as e:
                print(f"SUCCESS: Template was rejected with error: {e}")

    try:
        run_test()
        print("\n" + "=" * 60)
        print("Test FAILED: Templates with duplicate else were accepted")
    except AssertionError as e:
        print(f"Test FAILED: Template with duplicate else was accepted when it should have been rejected")
        # Show what actually happens
        content = "{{if False}}\nA\n{{else}}0\n{{else}}1\n{{endif}}"
        print(f"\nActual behavior with template: {repr(content)}")
        template = Template(content)
        result = template.substitute({})
        print(f"Result: {repr(result)}")
        print("\nThis demonstrates that invalid syntax is silently accepted.")
    except Exception as e:
        print(f"Unexpected error: {e}")
```

<details>

<summary>
**Failing input**: Templates with 2+ else clauses or elif after else
</summary>
```
Running property-based test for duplicate else clauses...
============================================================

Testing with 2 else clauses:
Template content: '{{if False}}\nA\n{{else}}0\n{{else}}1\n{{endif}}'
ERROR: Template was accepted (should have raised TemplateError)
Result when executed: '0\n'

Testing with 3 else clauses:
Template content: '{{if False}}\nA\n{{else}}0\n{{else}}1\n{{else}}2\n{{endif}}'
ERROR: Template was accepted (should have raised TemplateError)
Result when executed: '0\n'

Testing with 4 else clauses:
Template content: '{{if False}}\nA\n{{else}}0\n{{else}}1\n{{else}}2\n{{else}}3\n{{endif}}'
ERROR: Template was accepted (should have raised TemplateError)
Result when executed: '0\n'

============================================================
Test FAILED: Templates with duplicate else were accepted
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test case 1: Duplicate else clauses
print("Test 1: Duplicate else clauses")
print("-" * 40)
content1 = "{{if x}}A{{else}}B{{else}}C{{endif}}"
try:
    template1 = Template(content1)
    print(f"Template created successfully (should have failed)")
    print(f"When x=False: {template1.substitute({'x': False})}")
    print(f"When x=True: {template1.substitute({'x': True})}")
except Exception as e:
    print(f"Error (expected): {e}")

print("\nTest 2: elif after else")
print("-" * 40)
content2 = "{{if x}}A{{else}}B{{elif y}}C{{endif}}"
try:
    template2 = Template(content2)
    print(f"Template created successfully (should have failed)")
    print(f"When x=False, y=True: {template2.substitute({'x': False, 'y': True})}")
    print(f"When x=False, y=False: {template2.substitute({'x': False, 'y': False})}")
    print(f"When x=True, y=True: {template2.substitute({'x': True, 'y': True})}")
except Exception as e:
    print(f"Error (expected): {e}")

print("\nTest 3: Multiple duplicate else clauses")
print("-" * 40)
content3 = "{{if x}}A{{else}}B{{else}}C{{else}}D{{else}}E{{endif}}"
try:
    template3 = Template(content3)
    print(f"Template created successfully (should have failed)")
    print(f"When x=False: {template3.substitute({'x': False})}")
    print(f"When x=True: {template3.substitute({'x': True})}")
except Exception as e:
    print(f"Error (expected): {e}")
```

<details>

<summary>
Templates with invalid syntax are accepted without error
</summary>
```
Test 1: Duplicate else clauses
----------------------------------------
Template created successfully (should have failed)
When x=False: B
When x=True: A

Test 2: elif after else
----------------------------------------
Template created successfully (should have failed)
When x=False, y=True: B
When x=False, y=False: B
When x=True, y=True: A

Test 3: Multiple duplicate else clauses
----------------------------------------
Template created successfully (should have failed)
When x=False: B
When x=True: A
```
</details>

## Why This Is A Bug

This violates fundamental programming language syntax rules that are universally enforced across languages like Python, JavaScript, C, Java, and others. Specifically:

1. **Duplicate else clauses are invalid syntax**: No mainstream programming language allows multiple `else` clauses in a single conditional block. Python raises `SyntaxError: invalid syntax` for duplicate else.

2. **elif after else is invalid syntax**: The else clause must be the final clause in a conditional chain. Python raises `SyntaxError: invalid syntax` when elif follows else.

3. **Silent acceptance creates confusion**: The parser accepts these invalid constructs without error, but only executes the first matching clause. This means code like `{{else}}important_action{{else}}other_action` silently ignores `other_action`, which could lead to hard-to-debug template issues.

4. **Author acknowledgment**: The comment on line 291 of `_tempita.py` explicitly states `# @@: if/else/else gets through`, indicating the author was aware this invalid syntax passes through the parser when it shouldn't.

5. **Violates Tempita's Python-like syntax promise**: Tempita implements Python-like conditionals but doesn't enforce Python's syntax validation rules, creating an inconsistency that violates user expectations.

## Relevant Context

The bug exists in the `parse_cond()` function in `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py` starting at line 800. The function collects all conditional clauses without validating their structure:

- No tracking of whether an `else` clause has already been seen
- No check preventing `elif` after `else`
- All clauses are blindly appended to a list

During execution, the interpreter (`cond()` function) stops at the first matching clause, which masks but doesn't fix the invalid syntax acceptance.

The Tempita documentation shows standard if/elif/else examples but doesn't explicitly state that duplicate else or elif-after-else are prohibited, leaving this as an undocumented deviation from expected behavior.

GitHub repository: https://github.com/cython/cython (Tempita is bundled within Cython)
Affected file: `Cython/Tempita/_tempita.py` lines 800-813

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
@@ -809,6 +810,18 @@ def parse_cond(tokens, name, context):
         if (isinstance(tokens[0], tuple)
                 and tokens[0][0] == 'endif'):
             return ('cond', start) + tuple(pieces), tokens[1:]
+        # Validate conditional structure before parsing
+        if isinstance(tokens[0], tuple):
+            clause_type = tokens[0][0]
+            if clause_type == 'else':
+                if seen_else:
+                    raise TemplateError(
+                        'Duplicate else clause in conditional',
+                        position=tokens[0][1], name=name)
+                seen_else = True
+            elif clause_type.startswith('elif ') and seen_else:
+                raise TemplateError(
+                    'elif cannot appear after else',
+                    position=tokens[0][1], name=name)
         next_chunk, tokens = parse_one_cond(tokens, name, context)
         pieces.append(next_chunk)
```