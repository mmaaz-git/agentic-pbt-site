# Bug Report: Cython.Tempita.bunch Missing __delattr__ Implementation

**Target**: `Cython.Tempita.bunch`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `bunch` class in Cython.Tempita implements an incomplete Python attribute protocol - it supports setting and getting attributes but raises a misleading AttributeError when attempting to delete attributes, violating the expected behavior of Python objects.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Hypothesis test for Cython.Tempita.bunch delattr bug."""

import keyword
from hypothesis import given, strategies as st
import pytest
import Cython.Tempita as tempita

RESERVED = {"if", "for", "endif", "endfor", "else", "elif", "py", "default", "inherit"} | set(keyword.kwlist)
valid_identifier = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=10
).filter(lambda s: s not in RESERVED and s.isidentifier())


@given(valid_identifier, st.integers())
def test_bunch_delattr_not_supported(attr_name, value):
    b = tempita.bunch(**{attr_name: value})

    assert hasattr(b, attr_name)
    assert getattr(b, attr_name) == value

    with pytest.raises(AttributeError):
        delattr(b, attr_name)

if __name__ == "__main__":
    # Run the test
    test_bunch_delattr_not_supported()
```

<details>

<summary>
**Failing input**: `attr_name='a', value=0`
</summary>
```
============================= test session starts ==============================
platform linux -- Python 3.13.2, pytest-8.4.1, pluggy-1.5.0 -- /home/npc/miniconda/bin/python3
cachedir: .pytest_cache
hypothesis profile 'default'
rootdir: /home/npc/pbt/agentic-pbt/worker_/57
plugins: anyio-4.9.0, hypothesis-6.139.1, asyncio-1.2.0, langsmith-0.4.29
asyncio: mode=Mode.STRICT, debug=False, asyncio_default_fixture_loop_scope=None, asyncio_default_test_loop_scope=function
collecting ... collected 1 item

hypo.py::test_bunch_delattr_not_supported PASSED                         [100%]

============================== 1 passed in 0.16s ===============================
Test with attr_name="a", value=0
Initial: b.a=0
hasattr(b, "a")=True
delattr raised AttributeError: 'bunch' object has no attribute 'a'
After failed delattr: b.a=0
After failed delattr: hasattr(b, "a")=True
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of Cython.Tempita.bunch delattr bug."""

import Cython.Tempita as tempita

# Create a bunch object with some initial attributes
b = tempita.bunch(x=1, y=2)

# Verify initial attributes work
print(f"Initial state: b.x={b.x}, b.y={b.y}")
assert b.x == 1
assert b.y == 2

# Verify setattr works (modify existing attribute)
b.x = 10
print(f"After setattr: b.x={b.x}")
assert b.x == 10

# Verify setattr works (add new attribute)
setattr(b, 'z', 100)
print(f"After adding z: b.z={b.z}")
assert b.z == 100

# Verify getattr works
val = getattr(b, 'y')
print(f"Using getattr: getattr(b, 'y')={val}")
assert val == 2

# Check that the attribute exists before deletion
print(f"Has attribute 'x' before deletion: {hasattr(b, 'x')}")
assert hasattr(b, 'x')

# Try to delete attribute using delattr (THIS SHOULD FAIL)
print("\nAttempting to delete attribute 'x' using delattr...")
try:
    delattr(b, 'x')
    print("SUCCESS: Attribute deleted")
except AttributeError as e:
    print(f"FAILED: AttributeError raised: {e}")
    print(f"Has attribute 'x' after failed deletion: {hasattr(b, 'x')}")
    print(f"Value of b.x after failed deletion: {b.x}")
```

<details>

<summary>
AttributeError: 'bunch' object has no attribute 'x'
</summary>
```
Initial state: b.x=1, b.y=2
After setattr: b.x=10
After adding z: b.z=100
Using getattr: getattr(b, 'y')=2
Has attribute 'x' before deletion: True

Attempting to delete attribute 'x' using delattr...
FAILED: AttributeError raised: 'bunch' object has no attribute 'x'
Has attribute 'x' after failed deletion: True
Value of b.x after failed deletion: 10
```
</details>

## Why This Is A Bug

The `bunch` class violates Python's attribute protocol contract in several ways:

1. **Inconsistent API**: The class implements `__setattr__` and `__getattr__` but not `__delattr__`, creating an asymmetric interface where attributes can be created, read, and modified but not deleted.

2. **Misleading Error Message**: When attempting to delete an existing attribute, the error message states "'bunch' object has no attribute 'x'" even though the attribute clearly exists (can be read and modified).

3. **Violation of Duck Typing**: Python code that expects standard attribute behavior will fail when working with `bunch` objects, breaking compatibility with:
   - Object pooling/cleanup utilities
   - Testing frameworks that reset object state
   - Dynamic attribute management tools
   - Generic serialization/deserialization libraries

4. **Breaks Python's Data Model**: According to Python's data model, objects that support attribute assignment (`__setattr__`) typically should also support attribute deletion (`__delattr__`) unless attributes are meant to be immutable (which they aren't in `bunch`).

## Relevant Context

The `bunch` class is defined in `/Cython/Tempita/_tempita.py` (lines 393-421) as a subclass of `dict`. The implementation stores attributes as dictionary items using `__setattr__` to set `self[name] = value` and `__getattr__` to return `self[name]`.

Since `bunch` inherits from `dict`, the default `__delattr__` from `object` attempts to delete from the instance's `__dict__`, but `bunch` stores attributes as dictionary items (keys) rather than in `__dict__`. This causes the deletion to fail with a misleading error.

**Workarounds available:**
- Use dictionary methods: `del b['x']` or `b.pop('x')`
- Access the object as a dict rather than using attribute syntax

**Source code location**: [Cython/Tempita/_tempita.py:393-421](https://github.com/cython/cython/blob/master/Cython/Tempita/_tempita.py#L393-L421)

## Proposed Fix

```diff
--- a/Cython/Tempita/_tempita.py
+++ b/Cython/Tempita/_tempita.py
@@ -405,6 +405,13 @@ class bunch(dict):
         except KeyError:
             raise AttributeError(name)

+    def __delattr__(self, name):
+        try:
+            del self[name]
+        except KeyError:
+            raise AttributeError(
+                f"'bunch' object has no attribute '{name}'"
+            )
+
     def __getitem__(self, key):
         if 'default' in self:
             try:
```