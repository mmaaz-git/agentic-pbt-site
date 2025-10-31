# Bug Report: Cython.Tempita Template.substitute Namespace Override

**Target**: `Cython.Tempita.Template.substitute`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

Template namespace values incorrectly override substitute() arguments, reversing the expected precedence where runtime values should take priority over defaults.

## Property-Based Test

```python
@given(st.text(alphabet=string.ascii_letters + '_', min_size=1).filter(str.isidentifier),
       st.integers(min_value=0, max_value=100),
       st.integers(min_value=101, max_value=200))
def test_substitute_args_override_namespace(var_name, namespace_value, substitute_value):
    content = f"{{{{{var_name}}}}}"
    template = Template(content, namespace={var_name: namespace_value})
    result = template.substitute({var_name: substitute_value})

    assert result == str(substitute_value)
```

**Failing input**: Any template with namespace and conflicting substitute argument, e.g., `namespace={'x': 100}` with `substitute({'x': 200})`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Tempita import Template

template = Template('{{x}}', namespace={'x': 'namespace_value'})
result = template.substitute({'x': 'substitute_value'})

print(f"Result: {result}")
print(f"Expected: substitute_value")
print(f"Actual: {result}")
```

Output:
```
Result: namespace_value
Expected: substitute_value
Actual: namespace_value
```

## Why This Is A Bug

The Template constructor's docstring describes namespace as "default namespace" (line 22), indicating it should provide default values. In standard template library semantics, runtime values passed to substitute() should override defaults.

However, in the substitute() method (lines 182-186):

```python
ns = kw
ns['__template_name__'] = self.name
if self.namespace:
    ns.update(self.namespace)  # Bug: overwrites kw values
```

The `ns.update(self.namespace)` call overwrites any conflicting values from `kw` with namespace values, inverting the expected precedence.

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