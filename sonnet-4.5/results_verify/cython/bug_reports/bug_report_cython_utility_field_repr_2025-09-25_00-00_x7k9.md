# Bug Report: Cython.Utility Field.__repr__ Attribute Name Mismatch

**Target**: `Cython.Utility.Field.__repr__`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `Field.__repr__` method uses `kwonly` in its format string, but the actual attribute name is `kw_only` (with underscore), creating inconsistent naming in the repr output.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')
from Cython.Utility import field

@given(st.booleans())
def test_field_repr_consistency(kw_only_value):
    f = field(kw_only=kw_only_value)
    repr_str = repr(f)
    assert f'kw_only={kw_only_value!r}' in repr_str or f'kwonly={kw_only_value!r}' in repr_str
```

**Failing input**: Any value for `kw_only` parameter

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Utility import field

f = field(kw_only=True)
print(repr(f))
```

Output shows `kwonly=True` instead of `kw_only=True`, which is inconsistent with the attribute name.

## Why This Is A Bug

The Field class consistently uses `kw_only` (with underscore) throughout:
- Constructor parameter: `kw_only` (line 102 of Dataclasses.py)
- Attribute assignment: `self.kw_only` (line 52, 112)
- But `__repr__` uses: `'kwonly={!r}'` (line 66)

This creates confusion as the repr output doesn't match the actual attribute name.

## Fix

```diff
--- a/Cython/Utility/Dataclasses.py
+++ b/Cython/Utility/Dataclasses.py
@@ -63,7 +63,7 @@ class Field:
                 'hash={!r},'
                 'compare={!r},'
                 'metadata={!r},'
-                'kwonly={!r},'
+                'kw_only={!r},'
                 ')'.format(self.name, self.type, self.default,
                            self.default_factory, self.init,
                            self.repr, self.hash, self.compare,
```