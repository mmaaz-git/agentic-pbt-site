# Bug Report: Cython.Tempita Template.substitute Mutates Input Dictionary

**Target**: `Cython.Tempita.Template.substitute`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Template.substitute()` method mutates the input dictionary by adding internal keys and allowing template code to modify the namespace. This violates the principle that functions should not have unexpected side effects on their arguments.

## Property-Based Test

```python
@given(st.dictionaries(
    keys=st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
    values=st.integers(),
    min_size=0, max_size=5
))
def test_substitute_doesnt_mutate_input(vars_dict):
    assume('__template_name__' not in vars_dict)

    template = Template("{{x}}", name="test")
    original_keys = set(vars_dict.keys())

    result = template.substitute(vars_dict)

    assert set(vars_dict.keys()) == original_keys, \
        f"substitute() mutated input dict: added {set(vars_dict.keys()) - original_keys}"
```

**Failing input**: Any dictionary passed to `substitute()`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

template = Template("Value: {{x}}", name="test.txt")
context = {'x': 42}

print("Before:", list(context.keys()))
result = template.substitute(context)
print("After:", list(context.keys()))
```

Output:
```
Before: ['x']
After: ['x', '__template_name__']
```

The input dictionary is mutated to include `__template_name__`.

## Why This Is A Bug

Line 182 in `Template.substitute()` uses `ns = kw`, creating an alias rather than a copy. Line 183 then modifies `ns`, which also modifies the caller's dictionary. Any template code that modifies variables (like `{{py:x=1}}`) will also mutate the input dict.

This breaks the expectation that calling a method won't modify input arguments. It can cause surprising bugs when the same dictionary is reused across multiple template substitutions.

The code itself shows awareness of this issue: `_interpret_inherit()` at line 215 uses `ns = ns.copy()` to avoid mutations. But the main `substitute()` method doesn't apply the same protection.

## Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -179,7 +179,7 @@ class Template:
                     "If you pass in a single argument, you must pass in a dictionary-like object (with a .items() method); you gave %r"
                     % (args[0],))
             kw = args[0]
-        ns = kw
+        ns = kw.copy() if hasattr(kw, 'copy') else dict(kw)
         ns['__template_name__'] = self.name
         if self.namespace:
             ns.update(self.namespace)
```