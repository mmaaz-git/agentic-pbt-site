# Bug Report: pydantic.plugin.filter_handlers AttributeError on Missing __module__

**Target**: `pydantic.plugin._schema_validator.filter_handlers`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `filter_handlers` function crashes with `AttributeError` when a handler method lacks a `__module__` attribute (e.g., when the method is set to a builtin type instance like `int`, `str`, `list`, etc.).

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

import pytest
from hypothesis import given, settings, strategies as st
from unittest.mock import Mock
from pydantic.plugin._schema_validator import filter_handlers


@given(
    st.one_of(
        st.integers(),
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
        st.lists(st.integers()),
        st.dictionaries(st.text(), st.integers()),
        st.none(),
        st.booleans(),
    )
)
@settings(max_examples=500)
def test_filter_handlers_with_objects_without_module(obj):
    handler = Mock()
    setattr(handler, 'test_method', obj)

    try:
        result = filter_handlers(handler, 'test_method')
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
    except AttributeError:
        if hasattr(obj, '__module__'):
            pytest.fail(f"filter_handlers raised AttributeError for object with __module__: {type(obj)}")
        else:
            pytest.fail(f"BUG: filter_handlers crashes on objects without __module__ attribute (type: {type(obj).__name__})")
```

**Failing input**: `obj=0`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from unittest.mock import Mock
from pydantic.plugin._schema_validator import filter_handlers

handler = Mock()
handler.on_enter = 0

result = filter_handlers(handler, 'on_enter')
```

Output:
```
AttributeError: 'int' object has no attribute '__module__'. Did you mean: '__mod__'?
```

## Why This Is A Bug

The `filter_handlers` function is designed to filter out handler methods that are not implemented by a plugin directly. It does this by checking if `handler.__module__ == 'pydantic.plugin'` at line 135 of `_schema_validator.py`. However, it doesn't first verify that the `handler` object has a `__module__` attribute.

While handler methods should normally be callables (functions/methods) that have `__module__`, a malformed plugin implementation could accidentally set a handler attribute to a non-callable value (e.g., `int`, `str`, `list`, `dict`). These builtin type instances don't have a `__module__` attribute, causing an `AttributeError`.

This violates the expectation that `filter_handlers` should return a boolean value indicating whether to filter the handler, not crash with an obscure error about a missing `__module__` attribute.

## Fix

```diff
--- a/pydantic/plugin/_schema_validator.py
+++ b/pydantic/plugin/_schema_validator.py
@@ -132,7 +132,7 @@ def filter_handlers(handler_cls: BaseValidateHandlerProtocol, method_name: str)
     handler = getattr(handler_cls, method_name, None)
     if handler is None:
         return False
-    elif handler.__module__ == 'pydantic.plugin':
+    elif getattr(handler, '__module__', None) == 'pydantic.plugin':
         return False
     else:
         return True
```