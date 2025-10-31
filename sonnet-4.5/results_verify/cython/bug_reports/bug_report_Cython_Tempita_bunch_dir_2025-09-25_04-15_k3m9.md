# Bug Report: Cython.Tempita.bunch dir() Missing Attributes

**Target**: `Cython.Tempita.bunch`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `bunch` class does not include its dynamically set attributes in the `dir()` output, violating Python's introspection protocol and breaking IDE autocomplete, debuggers, and other introspection tools.

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

**Failing input**: `kwargs={'a': 0}`

## Reproducing the Bug

```python
import Cython.Tempita as tempita

b = tempita.bunch(x=1, y=2, z=3)

assert b.x == 1
assert hasattr(b, 'x')

assert 'x' not in dir(b)
```

## Why This Is A Bug

Python's `dir()` function is documented to return a list of valid attributes for an object. The `bunch` class supports attribute access via `getattr()` and dot notation, but these attributes are missing from `dir()` output. This violates Python's object introspection protocol and breaks:

1. IDE autocomplete features
2. Interactive debugging tools
3. Code that uses `dir()` to discover object attributes
4. Documentation generation tools

Since `bunch` supports `getattr(b, 'x')` and `b.x`, these attributes should appear in `dir(b)`.

## Fix

The `bunch` class needs to implement a `__dir__()` method that returns the dynamically set attributes. This would require tracking which attributes have been set and including them in the directory listing.

```python
def __dir__(self):
    base_attrs = object.__dir__(self)
    dynamic_attrs = [attr for attr in self.__class__._attrs if hasattr(self, attr)]
    return sorted(set(base_attrs + dynamic_attrs))
```

Note: The exact implementation depends on how `bunch` stores its attributes internally in the Cython code.