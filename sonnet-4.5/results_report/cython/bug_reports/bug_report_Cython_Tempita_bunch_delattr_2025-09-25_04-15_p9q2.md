# Bug Report: Cython.Tempita.bunch delattr Not Supported

**Target**: `Cython.Tempita.bunch`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `bunch` class supports `setattr()` to modify attributes but does not support `delattr()` to remove them, creating an inconsistent and incomplete implementation of Python's attribute protocol.

## Property-Based Test

```python
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
```

**Failing input**: `attr_name='a', value=0`

## Reproducing the Bug

```python
import Cython.Tempita as tempita

b = tempita.bunch(x=1, y=2)

assert b.x == 1

b.x = 10
assert b.x == 10

delattr(b, 'x')
```

Output:
```
AttributeError: 'bunch' object has no attribute 'x'
```

## Why This Is A Bug

Python's attribute protocol states that if an object supports `setattr()`, it should also support `delattr()` unless there's a specific reason to make attributes immutable. The `bunch` class allows:

1. Setting attributes at creation: `bunch(x=1)`
2. Reading attributes: `b.x` or `getattr(b, 'x')`
3. Modifying attributes: `b.x = 2` or `setattr(b, 'x', 2)`

But it does NOT allow:
4. Deleting attributes: `del b.x` or `delattr(b, 'x')`

This inconsistency violates the principle of least surprise and would affect code that expects standard Python object behavior, such as:
- Object pooling/cleanup code
- Testing frameworks that need to reset object state
- Serialization libraries
- Dynamic attribute management utilities

## Fix

The `bunch` class needs to implement `__delattr__()` to support attribute deletion. This would require removing the attribute from whatever internal storage mechanism `bunch` uses.

```python
def __delattr__(self, name):
    if name in self._internal_attrs:
        del self._internal_attrs[name]
    else:
        raise AttributeError(f"'bunch' object has no attribute '{name}'")
```

Note: The exact implementation depends on how `bunch` stores its attributes internally in the Cython code.