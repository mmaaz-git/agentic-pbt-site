# Bug Report: Cython.Tempita.Template.substitute Namespace Override Precedence

**Target**: `Cython.Tempita.Template.substitute`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Template namespace values incorrectly override substitute() arguments, reversing the expected precedence where runtime values should take priority over defaults.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
import string
from Cython.Tempita import Template

@given(st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
       st.integers(min_value=0, max_value=100),
       st.integers(min_value=101, max_value=200))
def test_substitute_args_override_namespace(var_name, namespace_value, substitute_value):
    content = f"{{{{{var_name}}}}}"
    template = Template(content, namespace={var_name: namespace_value})
    result = template.substitute({var_name: substitute_value})

    assert result == str(substitute_value), f"Expected {substitute_value}, got {result}"

# Run the test
test_substitute_args_override_namespace()
```

<details>

<summary>
**Failing input**: `var_name='A', namespace_value=0, substitute_value=101`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 19, in <module>
    test_substitute_args_override_namespace()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 9, in test_substitute_args_override_namespace
    st.integers(min_value=0, max_value=100),
            ^^^
  File "/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/51/hypo.py", line 16, in test_substitute_args_override_namespace
    assert result == str(substitute_value), f"Expected {substitute_value}, got {result}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 101, got 0
Falsifying example: test_substitute_args_override_namespace(
    # The test always failed when commented parts were varied together.
    var_name='A',  # or any other generated value
    namespace_value=0,  # or any other generated value
    substitute_value=101,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

# Test case 1: String values
print("Test 1: String values")
template = Template('{{x}}', namespace={'x': 'namespace_value'})
result = template.substitute({'x': 'substitute_value'})
print(f"Result: {result}")
print(f"Expected: substitute_value")
print(f"Actual: {result}")
print()

# Test case 2: Integer values
print("Test 2: Integer values")
template = Template('{{x}}', namespace={'x': 100})
result = template.substitute({'x': 200})
print(f"Result: {result}")
print(f"Expected: 200")
print(f"Actual: {result}")
print()

# Test case 3: Multiple variables
print("Test 3: Multiple variables")
template = Template('{{x}} {{y}}', namespace={'x': 'default_x', 'y': 'default_y'})
result = template.substitute({'x': 'new_x', 'y': 'new_y'})
print(f"Result: {result}")
print(f"Expected: new_x new_y")
print(f"Actual: {result}")
```

<details>

<summary>
Namespace values override substitute arguments in all test cases
</summary>
```
Test 1: String values
Result: namespace_value
Expected: substitute_value
Actual: namespace_value

Test 2: Integer values
Result: 100
Expected: 200
Actual: 100

Test 3: Multiple variables
Result: default_x default_y
Expected: new_x new_y
Actual: default_x default_y
```
</details>

## Why This Is A Bug

The bug violates the documented and expected behavior of the Template class. The Template constructor's docstring (lines 21-24 of _tempita.py) explicitly states that namespace is a "**default namespace**", and references Python's `string.Template` as a behavioral model. The term "default" universally means values used when no other value is provided - runtime arguments should override defaults.

However, the current implementation in the `substitute()` method (lines 182-186) does the opposite:

```python
ns = kw                          # ns references the substitute arguments
ns['__template_name__'] = self.name
if self.namespace:
    ns.update(self.namespace)    # BUG: overwrites substitute values with namespace
```

The `ns.update(self.namespace)` call overwrites any conflicting values from the substitute() arguments with namespace values. Since `ns` is just a reference to `kw` (not a copy), this directly modifies the input dictionary and uses namespace values instead of the provided substitute values.

This violates three key expectations:
1. **Documentation contract**: The namespace is described as "default" but acts as an override
2. **Standard precedence**: In Python's string.Template (the referenced model), keyword arguments take precedence over mapping arguments
3. **Programming conventions**: Default values should never override explicitly provided runtime values

## Relevant Context

The bug affects all Cython users who rely on the Tempita templating system for code generation. This is particularly problematic because:

- Users expect to set up templates with default values and override them at runtime
- The current behavior makes it impossible to override namespace values, defeating the purpose of having defaults
- The bug is systematic - it affects every template substitution where namespace and substitute arguments conflict

Related documentation:
- Cython.Tempita source: `/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages/Cython/Tempita/_tempita.py`
- Python string.Template docs: https://docs.python.org/3/library/string.html#template-strings (establishes precedence expectations)

## Proposed Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -179,10 +179,11 @@ class Template:
                     "If you pass in a single argument, you must pass in a dictionary-like object (with a .items() method); you gave %r"
                     % (args[0],))
             kw = args[0]
-        ns = kw
+        ns = self.namespace.copy() if self.namespace else {}
+        ns.update(kw)
         ns['__template_name__'] = self.name
-        if self.namespace:
-            ns.update(self.namespace)
         result, defs, inherit = self._interpret(ns)
         if not inherit:
             inherit = self.default_inherit
```