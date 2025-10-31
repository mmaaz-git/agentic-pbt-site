# Bug Report: pydantic.plugin build_wrapper on_error Handler Exception Suppression

**Target**: `pydantic.plugin._schema_validator.build_wrapper`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When an `on_error` handler raises an exception while handling a `ValidationError`, the handler's exception replaces the original `ValidationError`, causing it to be lost. This breaks the guarantee that validation errors are always propagated to the caller.

## Property-Based Test

```python
from unittest.mock import Mock
from hypothesis import given, strategies as st, settings
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError


@given(st.text(), st.text())
@settings(max_examples=20)
def test_on_error_handler_exception_suppresses_original(original_error_msg, handler_error_msg):
    def func():
        raise ValidationError.from_exception_data(original_error_msg, [])

    handler = Mock()

    def bad_on_error(error):
        raise RuntimeError(handler_error_msg)

    handler.on_error = bad_on_error

    wrapper = build_wrapper(func, [handler])

    try:
        wrapper()
        assert False, "Should have raised an exception"
    except RuntimeError as e:
        raise AssertionError(f"Original ValidationError was suppressed by handler exception")
    except ValidationError as e:
        pass
```

**Failing input**: Any values for `original_error_msg` and `handler_error_msg` trigger this bug.

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from unittest.mock import Mock
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError


def func():
    raise ValidationError.from_exception_data("Original validation error", [])


handler = Mock()

def bad_on_error(error):
    raise RuntimeError("Handler raised an exception")

handler.on_error = bad_on_error

wrapper = build_wrapper(func, [handler])

try:
    wrapper()
except RuntimeError as e:
    print(f"BUG: Caught RuntimeError from handler: {e}")
    print("The original ValidationError was suppressed!")
except ValidationError as e:
    print(f"Expected: Caught original ValidationError: {e}")
```

## Why This Is A Bug

The `build_wrapper` function is designed to call event handlers while propagating the original exception. However, if an `on_error` handler itself raises an exception, that exception replaces the original `ValidationError`. This violates the expected behavior where:

1. Validation errors should always be propagated to the caller
2. Plugin handlers should be observational and not affect control flow
3. Buggy handlers should not corrupt the validation process

This is especially problematic because it means a single misbehaving plugin can cause validation errors to be silently replaced with unrelated exceptions, making debugging extremely difficult.

## Fix

The fix should use exception chaining or a try-except block around handler calls to ensure the original exception is always re-raised:

```diff
--- a/pydantic/plugin/_schema_validator.py
+++ b/pydantic/plugin/_schema_validator.py
@@ -110,8 +110,14 @@ def build_wrapper(func: Callable[P, R], event_handlers: list[BaseValidateHandle
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
                 for on_exception_handler in on_exceptions:
```