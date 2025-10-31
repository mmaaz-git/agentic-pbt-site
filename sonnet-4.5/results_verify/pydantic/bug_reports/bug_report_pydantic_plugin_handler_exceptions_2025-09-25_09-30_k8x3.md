# Bug Report: pydantic.plugin Handler Exceptions Mask Validation Results

**Target**: `pydantic.plugin._schema_validator.build_wrapper()`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When plugin event handlers (`on_success`, `on_error`, `on_exception`) raise exceptions, they mask the original validation result or error, violating the principle that handlers should observe but not interfere with validation.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError
from unittest.mock import Mock

@given(st.integers())
def test_handler_exceptions_should_not_mask_success(value):
    def func(x):
        return x * 2

    handler = Mock()
    handler.on_success = Mock()
    handler.on_success.__module__ = 'test'
    handler.on_success.side_effect = RuntimeError("Handler failed")

    wrapped = build_wrapper(func, [handler])

    # BUG: This raises RuntimeError instead of returning func(value)
    # Expected: Result should be returned; handler failures should be logged/ignored
    result = wrapped(value)
    assert result == func(value)
```

**Failing input**: Any input triggers this bug when a handler raises an exception

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError
from unittest.mock import Mock

# Bug 1: on_success handler exceptions prevent return value
def successful_function(x):
    return x * 2

handler = Mock()
handler.on_success = Mock()
handler.on_success.__module__ = 'test'
handler.on_success.side_effect = RuntimeError("Handler crashed")

wrapped = build_wrapper(successful_function, [handler])

# Expected: returns 10
# Actual: raises RuntimeError
result = wrapped(5)  # Raises RuntimeError instead of returning 10

# Bug 2: on_error handler exceptions mask ValidationError
def failing_function(x):
    raise ValidationError.from_exception_data(
        "test",
        [{"type": "value_error", "loc": (), "msg": "invalid", "input": x}]
    )

error_handler = Mock()
error_handler.on_error = Mock()
error_handler.on_error.__module__ = 'test'
error_handler.on_error.side_effect = KeyError("Handler crashed")

wrapped2 = build_wrapper(failing_function, [error_handler])

# Expected: raises ValidationError
# Actual: raises KeyError
wrapped2(5)  # Raises KeyError instead of ValidationError
```

## Why This Is A Bug

The bug is in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py` lines 112-124:

**Current behavior**:
1. If an `on_success` handler raises an exception, it propagates and prevents the result from being returned (lines 120-123)
2. If an `on_error` handler raises an exception, it masks the original `ValidationError` (lines 112-115)
3. If an `on_exception` handler raises an exception, it masks the original exception (lines 116-119)

**Expected behavior**:
Plugin event handlers should observe validation but not interfere with its outcome. If a handler fails, it should be logged or suppressed, not allowed to mask the validation result.

**Impact**:
- A buggy plugin can cause successful validations to appear failed
- Original validation errors can be hidden, making debugging extremely difficult
- Breaks the contract that validation behavior is deterministic regardless of plugins

## Fix

Wrap handler calls in try-except blocks to prevent handler exceptions from interfering with validation:

```diff
diff --git a/_schema_validator.py b/_schema_validator.py
index 1234567..abcdefg 100644
--- a/_schema_validator.py
+++ b/_schema_validator.py
@@ -104,21 +104,36 @@ def build_wrapper(func: Callable[P, R], event_handlers: list[BaseValidateHandle

         @functools.wraps(func)
         def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
             for on_enter_handler in on_enters:
-                on_enter_handler(*args, **kwargs)
+                try:
+                    on_enter_handler(*args, **kwargs)
+                except Exception:
+                    pass  # or log the error

             try:
                 result = func(*args, **kwargs)
             except ValidationError as error:
                 for on_error_handler in on_errors:
-                    on_error_handler(error)
+                    try:
+                        on_error_handler(error)
+                    except Exception:
+                        pass  # or log the error
                 raise
             except Exception as exception:
                 for on_exception_handler in on_exceptions:
-                    on_exception_handler(exception)
+                    try:
+                        on_exception_handler(exception)
+                    except Exception:
+                        pass  # or log the error
                 raise
             else:
                 for on_success_handler in on_successes:
-                    on_success_handler(result)
+                    try:
+                        on_success_handler(result)
+                    except Exception:
+                        pass  # or log the error
                 return result

         return wrapper
```

Note: The fix shows `pass` for brevity, but production code should log these exceptions with appropriate context.