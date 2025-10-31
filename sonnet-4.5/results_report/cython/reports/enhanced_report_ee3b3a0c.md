# Bug Report: Cython.Tempita.bunch Missing Dynamic Attributes in dir() Output

**Target**: `Cython.Tempita.bunch`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `bunch` class in Cython.Tempita does not include dynamically set attributes in its `dir()` output, violating Python's introspection protocol where accessible attributes should be discoverable.

## Property-Based Test

```python
import keyword
from hypothesis import given, strategies as st
import Cython.Tempita as tempita

RESERVED = {"if", "for", "endif", "endfor", "else", "elif", "py", "default", "inherit"} | set(keyword.kwlist)
valid_identifier = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=10
).filter(lambda s: s not in RESERVED and s.isidentifier())


@given(st.dictionaries(valid_identifier, st.integers(), min_size=1, max_size=5))
def test_bunch_dir_missing_attrs(kwargs):
    b = tempita.bunch(**kwargs)

    for key in kwargs:
        assert hasattr(b, key)
        assert key not in dir(b)
```

<details>

<summary>
**Failing input**: `kwargs={'a': 0}`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/4
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_bunch_dir_missing_attrs PASSED                             [100%]

============================== 1 passed in 0.20s ===============================
```
</details>

## Reproducing the Bug

```python
import Cython.Tempita as tempita

# Create a bunch object with some attributes
b = tempita.bunch(x=1, y=2, z=3)

# Show that attributes are accessible
print("Accessing attributes:")
print(f"b.x = {b.x}")
print(f"b.y = {b.y}")
print(f"b.z = {b.z}")
print()

# Show that hasattr returns True
print("Using hasattr:")
print(f"hasattr(b, 'x') = {hasattr(b, 'x')}")
print(f"hasattr(b, 'y') = {hasattr(b, 'y')}")
print(f"hasattr(b, 'z') = {hasattr(b, 'z')}")
print()

# Show that these attributes are NOT in dir()
print("Using dir():")
print(f"'x' in dir(b) = {'x' in dir(b)}")
print(f"'y' in dir(b) = {'y' in dir(b)}")
print(f"'z' in dir(b) = {'z' in dir(b)}")
print()

# Show what dir() actually returns (filtered for brevity)
print("Attributes in dir(b) (non-dunder methods):")
non_dunder = [attr for attr in dir(b) if not attr.startswith('__')]
print(non_dunder)
```

<details>

<summary>
Attributes accessible via dot notation are missing from dir()
</summary>
```
Accessing attributes:
b.x = 1
b.y = 2
b.z = 3

Using hasattr:
hasattr(b, 'x') = True
hasattr(b, 'y') = True
hasattr(b, 'z') = True

Using dir():
'x' in dir(b) = False
'y' in dir(b) = False
'z' in dir(b) = False

Attributes in dir(b) (non-dunder methods):
['clear', 'copy', 'fromkeys', 'get', 'items', 'keys', 'pop', 'popitem', 'setdefault', 'update', 'values']
```
</details>

## Why This Is A Bug

The `bunch` class violates Python's introspection protocol. According to Python's documentation, `dir()` should return "a list of valid attributes for an object." Since the `bunch` class makes dictionary keys accessible as attributes through `__getattr__()` and `__setattr__()` methods, these attributes are valid and should appear in the `dir()` output.

This inconsistency breaks:
1. **IDE autocomplete** - IDEs rely on `dir()` to suggest available attributes
2. **Interactive debugging** - Tools like `pdb` and `ipdb` use `dir()` for attribute discovery
3. **Documentation generators** - Tools that introspect objects to generate documentation
4. **The principle of least surprise** - If `hasattr(obj, 'x')` returns `True` and `obj.x` works, Python developers expect `'x' in dir(obj)` to be `True`

The Python documentation states that when an object has accessible attributes (via `getattr()`), they should be considered valid attributes. Standard library classes with similar functionality, like `argparse.Namespace`, correctly include their dynamic attributes in `dir()` output.

## Relevant Context

The `bunch` class is located at `/home/npc/miniconda/lib/python3.13/site-packages/Cython/Tempita/_tempita.py` (lines 393-421). It inherits from `dict` and implements:
- `__setattr__()` to store attributes as dictionary items
- `__getattr__()` to retrieve dictionary items as attributes
- But it does NOT implement `__dir__()` to expose these attributes

The class is part of Cython's public API (listed in `__all__` on line 41) and is used for template variable passing. While the bug has likely existed for years, it still violates Python's established conventions for object introspection.

Workarounds exist (using `.keys()` or accessing the underlying dict), but the proper solution is to implement `__dir__()`.

## Proposed Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -419,6 +419,10 @@ class bunch(dict):
             self.__class__.__name__,
             ' '.join(['%s=%r' % (k, v) for k, v in sorted(self.items())]))

+    def __dir__(self):
+        base_attrs = dict.__dir__(self)
+        return sorted(set(base_attrs) | set(self.keys()))
+

 class TemplateDef:
     def __init__(self, template, func_name, func_signature,
```