# Bug Report: Cython.Tempita Template.substitute Mutates Input Dictionary

**Target**: `Cython.Tempita._tempita.Template.substitute`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `Template.substitute` method mutates the input dictionary by adding the `__template_name__` key, violating the principle of least surprise that method calls should not modify their arguments unless explicitly documented.

## Property-Based Test

```python
@given(st.dictionaries(
    keys=st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
    values=st.integers(),
    min_size=1, max_size=5
))
@settings(max_examples=100)
def test_substitute_does_not_mutate_input(input_dict):
    original_keys = set(input_dict.keys())

    content = "{{x}}" if 'x' in input_dict else "test"
    template = Template(content)
    result = template.substitute(input_dict)

    new_keys = set(input_dict.keys())
    added_keys = new_keys - original_keys

    assert added_keys == set(), f"substitute() should not mutate input dict"
```

**Failing input**: Any dictionary, e.g., `{'x': 42}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

template = Template("{{x}}", name="test.tmpl")
input_dict = {'x': 42}

print(f"Before: {input_dict}")
result = template.substitute(input_dict)
print(f"After:  {input_dict}")
print(f"Result: {result}")

if '__template_name__' in input_dict:
    print("\nBUG: Input dictionary was mutated!")
    print(f"Added key '__template_name__' with value: {input_dict['__template_name__']}")
```

## Why This Is A Bug

Line 182 in `Cython/Tempita/_tempita.py` contains `ns = kw` which creates an alias to the input dictionary, not a copy. Line 183 then adds `ns['__template_name__'] = self.name`, which mutates the original dictionary passed by the caller.

Functions should not have unexpected side effects on their arguments. Users calling `template.substitute(my_dict)` would reasonably expect `my_dict` to remain unchanged after the call. This mutation can cause:

1. Unexpected keys appearing in user dictionaries
2. Issues when the same dict is reused for multiple substitutions
3. Confusion when debugging (dict contents change without explicit assignment)

The substitute method's documentation doesn't mention this mutation, making it an undocumented side effect.

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