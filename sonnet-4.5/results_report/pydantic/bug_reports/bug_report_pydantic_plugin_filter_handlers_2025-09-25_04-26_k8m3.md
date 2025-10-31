# Bug Report: pydantic.plugin filter_handlers AttributeError

**Target**: `pydantic.plugin._schema_validator.filter_handlers`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `filter_handlers` function in `pydantic.plugin._schema_validator` directly accesses the `__module__` attribute without checking if it exists, which can cause an `AttributeError` when handler methods are callable objects that lack this attribute.

## Property-Based Test

```python
from hypothesis import given, strategies as st, settings
from pydantic.plugin._schema_validator import filter_handlers


class CallableWithoutModule:
    def __call__(self):
        pass


class CustomHandler:
    pass


@given(st.sampled_from(['on_enter', 'on_success', 'on_error', 'on_exception']))
@settings(max_examples=100)
def test_filter_handlers_handles_all_callables(method_name):
    handler = CustomHandler()

    callable_obj = CallableWithoutModule()

    if hasattr(callable_obj, '__module__'):
        del callable_obj.__module__

    setattr(handler, method_name, callable_obj)

    result = filter_handlers(handler, method_name)
    assert isinstance(result, bool)
```

**Failing input**: Any handler with a method that is a callable object lacking `__module__` attribute

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from pydantic.plugin._schema_validator import filter_handlers


class CallableWithoutModule:
    def __call__(self):
        pass


class Handler:
    pass


handler = Handler()
handler.on_enter = CallableWithoutModule()

if hasattr(handler.on_enter, '__module__'):
    delattr(type(handler.on_enter), '__module__')

filter_handlers(handler, 'on_enter')
```

## Why This Is A Bug

The `filter_handlers` function on line 135 of `_schema_validator.py` directly accesses `handler.__module__`:

```python
elif handler.__module__ == 'pydantic.plugin':
```

This assumes all callable objects have a `__module__` attribute. While most Python callables do have this attribute, not all do:
- Callable instances (classes with `__call__`) might have `__module__` deleted or not set
- Some C extension methods
- Certain descriptors or dynamically created callables

**Evidence of inconsistency**: Other parts of the pydantic codebase use the safer `getattr(obj, '__module__', default)` pattern:
- `pydantic/v1/class_validators.py:151`: `getattr(f_cls.__func__, '__module__', '<No __module__>')`
- `pydantic/_internal/_model_construction.py:475`: `getattr(ann_type, '__module__', None)`
- `pydantic/_internal/_core_utils.py:81`: `getattr(origin, '__module__', '<No __module__>')`
- `pydantic/_internal/_namespace_utils.py:54`: `getattr(obj, '__module__', None)`

The `filter_handlers` function should follow this same safe pattern to avoid potential `AttributeError` exceptions.

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
         # this is the original handler, from the protocol due to runtime inheritance
         # we don't want to call it
         return False
```