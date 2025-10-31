# Bug Report: pydantic.plugin.filter_handlers AttributeError on Callables Without __module__

**Target**: `pydantic.plugin._schema_validator.filter_handlers`
**Severity**: Low
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `filter_handlers` function crashes with `AttributeError` when processing handler methods that are callable objects lacking a `__module__` attribute, despite other parts of pydantic safely handling this case.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from pydantic.plugin._schema_validator import filter_handlers


class CallableWithoutModule:
    def __call__(self):
        pass

    def __getattribute__(self, name):
        if name == '__module__':
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '__module__'")
        return object.__getattribute__(self, name)


class CustomHandler:
    pass


@given(st.sampled_from(['on_enter', 'on_success', 'on_error', 'on_exception']))
@settings(max_examples=100)
def test_filter_handlers_handles_all_callables(method_name):
    handler = CustomHandler()

    # Create a callable without __module__ attribute
    callable_obj = CallableWithoutModule()

    setattr(handler, method_name, callable_obj)

    # This should not raise an AttributeError
    result = filter_handlers(handler, method_name)
    assert isinstance(result, bool)


if __name__ == '__main__':
    test_filter_handlers_handles_all_callables()
```

<details>

<summary>
**Failing input**: `method_name='on_enter'`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 38, in <module>
    test_filter_handlers_handles_all_callables()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 23, in test_filter_handlers_handles_all_callables
    @settings(max_examples=100)
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 33, in test_filter_handlers_handles_all_callables
    result = filter_handlers(handler, method_name)
  File "/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py", line 135, in filter_handlers
    elif handler.__module__ == 'pydantic.plugin':
         ^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 14, in __getattribute__
    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '__module__'")
AttributeError: 'CallableWithoutModule' object has no attribute '__module__'
Falsifying example: test_filter_handlers_handles_all_callables(
    method_name='on_enter',
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from pydantic.plugin._schema_validator import filter_handlers


# Create a callable object without __module__ attribute
class CallableWithoutModule:
    def __call__(self):
        pass

    # Override __getattribute__ to simulate missing __module__
    def __getattribute__(self, name):
        if name == '__module__':
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '__module__'")
        return object.__getattribute__(self, name)


class Handler:
    pass


handler = Handler()
handler.on_enter = CallableWithoutModule()

# This will raise an AttributeError when filter_handlers tries to access __module__
result = filter_handlers(handler, 'on_enter')
print(f"Result: {result}")
```

<details>

<summary>
AttributeError when accessing __module__ on callable without the attribute
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/repo.py", line 27, in <module>
    result = filter_handlers(handler, 'on_enter')
  File "/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py", line 135, in filter_handlers
    elif handler.__module__ == 'pydantic.plugin':
         ^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/repo.py", line 15, in __getattribute__
    raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '__module__'")
AttributeError: 'CallableWithoutModule' object has no attribute '__module__'. Did you mean: '__reduce__'?
```
</details>

## Why This Is A Bug

The `filter_handlers` function in `pydantic/plugin/_schema_validator.py` line 135 directly accesses the `__module__` attribute without checking if it exists:

```python
elif handler.__module__ == 'pydantic.plugin':
```

This violates the principle of defensive programming that pydantic follows elsewhere in its codebase. While most Python callables have a `__module__` attribute, not all do. Examples include:
- Custom callable classes that override `__getattribute__`
- Certain C extensions that don't define `__module__`
- Dynamically created callables where `__module__` might be missing
- Objects where `__module__` has been explicitly deleted or not set

The function should gracefully handle callables without `__module__` attributes rather than crashing, especially since the purpose is just to filter out protocol-inherited methods.

## Relevant Context

The pydantic codebase consistently uses the safer `getattr(obj, '__module__', default)` pattern in other locations to avoid this exact issue:

- `pydantic/v1/class_validators.py:151`: `getattr(f_cls.__func__, '__module__', '<No __module__>')`
- `pydantic/_internal/_model_construction.py:475`: `getattr(ann_type, '__module__', None)`
- `pydantic/_internal/_core_utils.py:81`: `getattr(origin, '__module__', '<No __module__>')`
- `pydantic/_internal/_namespace_utils.py:54`: `getattr(obj, '__module__', None)`

The `filter_handlers` function is used internally by the plugin system to determine which handler methods should be called. It filters out methods that are inherited from the base protocol (in the 'pydantic.plugin' module) versus those implemented by actual plugins.

## Proposed Fix

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