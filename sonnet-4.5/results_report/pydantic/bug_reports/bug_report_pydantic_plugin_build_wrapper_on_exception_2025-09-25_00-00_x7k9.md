# Bug Report: pydantic.plugin build_wrapper on_exception Handler Exception Suppression

**Target**: `pydantic.plugin._schema_validator.build_wrapper`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When an `on_exception` handler raises an exception while handling a non-ValidationError exception, the handler's exception replaces the original exception, causing it to be lost. This breaks exception transparency and makes debugging impossible.

## Property-Based Test

```python
from unittest.mock import Mock
from hypothesis import given, strategies as st, settings
from pydantic.plugin._schema_validator import build_wrapper


@given(st.text(), st.text())
@settings(max_examples=20)
def test_on_exception_handler_exception_suppresses_original(original_error_msg, handler_error_msg):
    def func():
        raise ValueError(original_error_msg)

    handler = Mock()

    def bad_on_exception(exception):
        raise RuntimeError(handler_error_msg)

    handler.on_exception = bad_on_exception

    wrapper = build_wrapper(func, [handler])

    try:
        wrapper()
        assert False, "Should have raised an exception"
    except RuntimeError as e:
        raise AssertionError(f"Original exception was suppressed by handler exception")
    except ValueError as e:
        pass
```

**Failing input**: Any values for `original_error_msg` and `handler_error_msg` trigger this bug.

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from unittest.mock import Mock
from pydantic.plugin._schema_validator import build_wrapper


def func():
    raise ValueError("Original error")


handler = Mock()

def bad_on_exception(exception):
    raise RuntimeError("Handler raised an exception")

handler.on_exception = bad_on_exception

wrapper = build_wrapper(func, [handler])

try:
    wrapper()
except RuntimeError as e:
    print(f"BUG: Caught RuntimeError from handler: {e}")
    print("The original ValueError was suppressed!")
except ValueError as e:
    print(f"Expected: Caught original ValueError: {e}")
```

## Why This Is A Bug

Similar to the `on_error` handler bug, when an `on_exception` handler raises an exception, it replaces the original exception. This violates the principle that plugin handlers should be observational and not affect the application's exception handling.

The consequences are severe:
1. Original exceptions from application code are lost
2. Debugging becomes impossible as the wrong exception is shown to users
3. A single buggy plugin can break the entire validation pipeline
4. Exception context and stack traces point to the handler instead of the actual error source

## Fix

Apply the same fix pattern as for `on_error` handlers - wrap handler calls in try-except blocks:

```diff
--- a/pydantic/plugin/_schema_validator.py
+++ b/pydantic/plugin/_schema_validator.py
@@ -110,12 +110,24 @@ def build_wrapper(func: Callable[P, R], event_handlers: list[BaseValidateHandle
             try:
                 result = func(*args, **kwargs)
             except ValidationError as error:
-                for on_error_handler in on_errors:
-                    on_error_handler(error)
+                for on_error_handler in on_errors:
+                    try:
+                        on_error_handler(error)
+                    except Exception as handler_exception:
+                        import warnings
+                        warnings.warn(
+                            f'Exception in plugin on_error handler: {handler_exception!r}'
+                        )
                 raise
             except Exception as exception:
-                for on_exception_handler in on_exceptions:
-                    on_exception_handler(exception)
+                for on_exception_handler in on_exceptions:
+                    try:
+                        on_exception_handler(exception)
+                    except Exception as handler_exception:
+                        import warnings
+                        warnings.warn(
+                            f'Exception in plugin on_exception handler: {handler_exception!r}'
+                        )
                 raise
             else:
                 for on_success_handler in on_successes:
```