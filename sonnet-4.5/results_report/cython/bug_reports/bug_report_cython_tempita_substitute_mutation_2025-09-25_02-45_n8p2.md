# Bug Report: Cython.Tempita Template.substitute Mutates Input Dictionary

**Target**: `Cython.Tempita.Template.substitute`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The substitute() method mutates the caller's input dictionary by adding internal keys like `__template_name__` and namespace values, violating the principle of least surprise and potentially causing bugs in caller code.

## Property-Based Test

```python
@given(st.dictionaries(
    keys=st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
    values=st.integers(),
    min_size=1, max_size=3
))
def test_substitute_does_not_mutate_input(user_vars):
    if not user_vars:
        user_vars = {'x': 1}

    var_to_use = list(user_vars.keys())[0]
    template = Template(f'{{{{{var_to_use}}}}}', namespace={'other': 999})
    original_vars = user_vars.copy()

    template.substitute(user_vars)

    assert user_vars == original_vars
```

**Failing input**: Any dictionary passed to substitute(), e.g., `{'x': 'value'}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

user_dict = {'x': 'value', 'y': 42}
print(f"Before: {user_dict}")

template = Template('{{x}}', namespace={'z': 100})
template.substitute(user_dict)

print(f"After:  {user_dict}")
print(f"\nAdded keys: {set(user_dict.keys()) - {'x', 'y'}}")
```

Output:
```
Before: {'x': 'value', 'y': 42}
After:  {'x': 'value', 'y': 42, '__template_name__': None, 'z': 100}

Added keys: {'__template_name__', 'z'}
```

## Why This Is A Bug

In the substitute() method (lines 182-186):

```python
ns = kw
ns['__template_name__'] = self.name
if self.namespace:
    ns.update(self.namespace)
```

Line 182 makes `ns` reference the same dict object as `kw` (the user's input). Subsequent modifications to `ns` (lines 183, 185) mutate the caller's dictionary, which is an unexpected side effect.

Standard library functions like `str.format()` and `string.Template.substitute()` do not mutate their input dictionaries. Users expect immutability for read-only operations.

## Fix

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