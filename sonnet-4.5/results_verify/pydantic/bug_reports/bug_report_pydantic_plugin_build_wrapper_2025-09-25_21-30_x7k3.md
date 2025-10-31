# Bug Report: pydantic.plugin build_wrapper Exception Masking

**Target**: `pydantic.plugin._schema_validator.build_wrapper`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When event handlers (on_error, on_success, on_exception) raise exceptions, they mask the original error/return value and prevent subsequent handlers from executing. This causes silent failures and breaks the plugin handler chain.

## Property-Based Test

```python
from unittest.mock import Mock
from hypothesis import given, strategies as st
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError


def test_first_on_error_exception_prevents_second_handler():
    handler1_called = []
    handler2_called = []

    def original_func(x):
        raise ValidationError.from_exception_data('test', [])

    handler1 = Mock()
    handler1.on_error = Mock(side_effect=RuntimeError("handler1 error"))
    handler1.on_error.__module__ = 'test1'

    handler2 = Mock()
    handler2.on_error = lambda e: handler2_called.append(True)
    handler2.on_error.__module__ = 'test2'

    wrapped = build_wrapper(original_func, [handler1, handler2])

    caught_exception = None
    try:
        wrapped(5)
    except Exception as e:
        caught_exception = e

    assert isinstance(caught_exception, RuntimeError)
    assert len(handler2_called) == 0
```

**Failing input**: Any input that triggers a handler exception

## Reproducing the Bug

```python
from unittest.mock import Mock
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError


def original_func(x):
    raise ValidationError.from_exception_data('validation_failed', [])


handler1 = Mock()
handler1.on_error = Mock(side_effect=RuntimeError("handler1 crashed"))
handler1.on_error.__module__ = 'plugin1'

handler2 = Mock()
handler2.on_error = Mock()
handler2.on_error.__module__ = 'plugin2'

wrapped = build_wrapper(original_func, [handler1, handler2])

try:
    wrapped(42)
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Expected: ValidationError")
    print(f"Actual: {type(e).__name__}")
    print(f"Handler2 called: {handler2.on_error.called}")
```

**Output:**
```
Exception type: RuntimeError
Expected: ValidationError
Actual: RuntimeError
Handler2 called: False
```

## Why This Is A Bug

The current implementation in `_schema_validator.py` lines 112-123 does not protect against handler exceptions:

```python
except ValidationError as error:
    for on_error_handler in on_errors:
        on_error_handler(error)  # If this raises, the loop breaks
    raise  # This is never reached if handler raises
```

This violates expected behavior because:
1. **Masks original errors**: A plugin bug causes the original ValidationError to be hidden
2. **Breaks handler chain**: Subsequent handlers never execute
3. **No error isolation**: One buggy plugin breaks all plugins
4. **Silent failures**: Users lose critical validation error information

## Fix

Wrap each handler call in try/except to ensure all handlers run and the original error is preserved:

```diff
--- a/pydantic/plugin/_schema_validator.py
+++ b/pydantic/plugin/_schema_validator.py
@@ -105,18 +105,30 @@ def build_wrapper(func: Callable[P, R], event_handlers: list[BaseValidateHandle
         @functools.wraps(func)
         def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
             for on_enter_handler in on_enters:
-                on_enter_handler(*args, **kwargs)
+                try:
+                    on_enter_handler(*args, **kwargs)
+                except Exception:
+                    pass

             try:
                 result = func(*args, **kwargs)
             except ValidationError as error:
                 for on_error_handler in on_errors:
-                    on_error_handler(error)
+                    try:
+                        on_error_handler(error)
+                    except Exception:
+                        pass
                 raise
             except Exception as exception:
                 for on_exception_handler in on_exceptions:
-                    on_exception_handler(exception)
+                    try:
+                        on_exception_handler(exception)
+                    except Exception:
+                        pass
                 raise
             else:
                 for on_success_handler in on_successes:
-                    on_success_handler(result)
+                    try:
+                        on_success_handler(result)
+                    except Exception:
+                        pass
                 return result

         return wrapper
```

Alternatively, consider logging handler exceptions instead of silently suppressing them.