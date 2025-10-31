# Bug Report: xarray.compat.npcompat.isdtype List Handling

**Target**: `xarray.compat.npcompat.isdtype`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The fallback implementation of `isdtype` (used when numpy < 2.0) crashes with a confusing `TypeError: unhashable type: 'list'` when passed a list of kinds instead of a tuple.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import numpy as np

kind_mapping = {
    "bool": np.bool_,
    "signed integer": np.signedinteger,
    "unsigned integer": np.unsignedinteger,
    "integral": np.integer,
    "real floating": np.floating,
    "complex floating": np.complexfloating,
    "numeric": np.number,
}

def isdtype_fallback(dtype, kind):
    kinds = kind if isinstance(kind, tuple) else (kind,)
    str_kinds = {k for k in kinds if isinstance(k, str)}
    type_kinds = {k.type for k in kinds if isinstance(k, np.dtype)}

    if unknown_kind_types := set(kinds) - str_kinds - type_kinds:
        raise TypeError(
            f"kind must be str, np.dtype or a tuple of these, got {unknown_kind_types}"
        )
    if unknown_kinds := {k for k in str_kinds if k not in kind_mapping}:
        raise ValueError(
            f"unknown kind: {unknown_kinds}, must be a np.dtype or one of {list(kind_mapping)}"
        )

    translated_kinds = {kind_mapping[k] for k in str_kinds} | type_kinds
    if isinstance(dtype, np.generic):
        return isinstance(dtype, translated_kinds)
    else:
        return any(np.issubdtype(dtype, k) for k in translated_kinds)

@given(st.sampled_from([np.int32, np.float64, np.bool_]))
def test_isdtype_accepts_list_of_kinds(dtype):
    try:
        result = isdtype_fallback(dtype, ["signed integer", "bool"])
    except TypeError as e:
        assert "list" not in str(e).lower() or "unhashable" not in str(e).lower(), \
            f"Should give clear error about list not being supported, not: {e}"
```

**Failing input**: `isdtype_fallback(np.int32, ["signed integer", "bool"])`

## Reproducing the Bug

```python
import numpy as np

kind_mapping = {
    "bool": np.bool_,
    "signed integer": np.signedinteger,
    "unsigned integer": np.unsignedinteger,
    "integral": np.integer,
    "real floating": np.floating,
    "complex floating": np.complexfloating,
    "numeric": np.number,
}

def isdtype_fallback(dtype, kind):
    kinds = kind if isinstance(kind, tuple) else (kind,)
    str_kinds = {k for k in kinds if isinstance(k, str)}
    type_kinds = {k.type for k in kinds if isinstance(k, np.dtype)}

    if unknown_kind_types := set(kinds) - str_kinds - type_kinds:
        raise TypeError(
            f"kind must be str, np.dtype or a tuple of these, got {unknown_kind_types}"
        )

    if unknown_kinds := {k for k in str_kinds if k not in kind_mapping}:
        raise ValueError(
            f"unknown kind: {unknown_kinds}, must be a np.dtype or one of {list(kind_mapping)}"
        )

    translated_kinds = {kind_mapping[k] for k in str_kinds} | type_kinds
    if isinstance(dtype, np.generic):
        return isinstance(dtype, translated_kinds)
    else:
        return any(np.issubdtype(dtype, k) for k in translated_kinds)

isdtype_fallback(np.int32, ["signed integer", "bool"])
```

## Why This Is A Bug

The code on line 55 converts non-tuple inputs to tuples using `kinds = kind if isinstance(kind, tuple) else (kind,)`. When `kind` is a list like `["signed integer", "bool"]`, this creates `kinds = (["signed integer", "bool"],)` - a tuple containing a single list.

Later, line 61 tries to compute `set(kinds) - str_kinds - type_kinds`, which fails because lists are unhashable and cannot be added to a set.

This violates user expectations in two ways:
1. Lists are a natural input type (similar to tuples) and users might reasonably pass them
2. The error message "unhashable type: 'list'" is confusing and doesn't indicate the actual problem

Note: This only affects users with numpy < 2.0. The numpy >= 2.0 implementation properly rejects lists with a clear error message.

## Fix

```diff
def isdtype(
    dtype: np.dtype[Any] | type[Any], kind: DTypeLike | tuple[DTypeLike, ...]
) -> bool:
-   kinds = kind if isinstance(kind, tuple) else (kind,)
+   if isinstance(kind, (list, tuple)):
+       kinds = tuple(kind)
+   else:
+       kinds = (kind,)
    str_kinds = {k for k in kinds if isinstance(k, str)}
    type_kinds = {k.type for k in kinds if isinstance(k, np.dtype)}

    if unknown_kind_types := set(kinds) - str_kinds - type_kinds:
        raise TypeError(
            f"kind must be str, np.dtype or a tuple of these, got {unknown_kind_types}"
        )
    if unknown_kinds := {k for k in str_kinds if k not in kind_mapping}:
        raise ValueError(
            f"unknown kind: {unknown_kinds}, must be a np.dtype or one of {list(kind_mapping)}"
        )

    translated_kinds = {kind_mapping[k] for k in str_kinds} | type_kinds
    if isinstance(dtype, np.generic):
        return isinstance(dtype, translated_kinds)
    else:
        return any(np.issubdtype(dtype, k) for k in translated_kinds)
```