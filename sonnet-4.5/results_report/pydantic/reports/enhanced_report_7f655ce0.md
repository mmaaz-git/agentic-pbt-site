# Bug Report: pydantic.plugin._schema_validator.build_wrapper Exception Masking in Event Handlers

**Target**: `pydantic.plugin._schema_validator.build_wrapper`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When plugin event handlers (on_error, on_success, on_exception, on_enter) raise exceptions, they completely mask the original errors/return values and prevent subsequent handlers in the chain from executing, causing silent failures and breaking plugin isolation.

## Property-Based Test

```python
"""Property-based test demonstrating pydantic plugin build_wrapper exception masking bug."""

from unittest.mock import Mock
from hypothesis import given, strategies as st
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError


@given(input_value=st.integers())
def test_first_on_error_exception_prevents_second_handler(input_value):
    """Test that when first on_error handler raises, second handler never executes."""
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
        wrapped(input_value)
    except Exception as e:
        caught_exception = e

    # Bug: The RuntimeError from handler1 masks the original ValidationError
    assert isinstance(caught_exception, RuntimeError)
    # Bug: handler2 never gets called because handler1 crashed
    assert len(handler2_called) == 0


if __name__ == "__main__":
    # Run the property-based test
    test_first_on_error_exception_prevents_second_handler()
```

<details>

<summary>
**Failing input**: `0` (any integer value triggers the bug)
</summary>
```
Test passed successfully - property-based test confirms the bug:
- When first on_error handler raises RuntimeError, it masks the original ValidationError
- Second handler never gets called because first handler crashed
- This violates expected plugin isolation behavior
```
</details>

## Reproducing the Bug

```python
"""Minimal reproduction of pydantic plugin build_wrapper exception masking bug."""

from unittest.mock import Mock
from pydantic.plugin._schema_validator import build_wrapper
from pydantic_core import ValidationError


def original_func(x):
    """Function that raises a ValidationError."""
    raise ValidationError.from_exception_data('validation_failed', [])


# Create two plugin handlers
handler1 = Mock()
handler1.on_error = Mock(side_effect=RuntimeError("handler1 crashed"))
handler1.on_error.__module__ = 'plugin1'

handler2 = Mock()
handler2.on_error = Mock()
handler2.on_error.__module__ = 'plugin2'

# Wrap the function with the handlers
wrapped = build_wrapper(original_func, [handler1, handler2])

# Execute and observe the behavior
try:
    wrapped(42)
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print(f"Expected exception type: ValidationError")
    print(f"Actual exception type: {type(e).__name__}")
    print(f"Handler2 called: {handler2.on_error.called}")
    print(f"\nThis demonstrates the bug:")
    print(f"1. The original ValidationError is masked by handler1's RuntimeError")
    print(f"2. Handler2 never gets executed because handler1 crashed")
```

<details>

<summary>
RuntimeError masks original ValidationError
</summary>
```
Exception type: RuntimeError
Exception message: handler1 crashed
Expected exception type: ValidationError
Actual exception type: RuntimeError
Handler2 called: False

This demonstrates the bug:
1. The original ValidationError is masked by handler1's RuntimeError
2. Handler2 never gets executed because handler1 crashed
```
</details>

## Why This Is A Bug

This violates expected behavior for plugin systems in multiple critical ways:

1. **Error Masking**: The original `ValidationError` containing important validation failure details is completely replaced by the plugin's `RuntimeError`. Users lose all information about what validation actually failed, receiving only "handler1 crashed" instead of the actual validation error message.

2. **Plugin Chain Breaking**: The plugin architecture implies multiple plugins can coexist, but one faulty plugin prevents all subsequent plugins from executing. In the code at lines 113-114 of `_schema_validator.py`, when `on_error_handler(error)` raises an exception, the loop terminates immediately and the `raise` on line 115 is never reached.

3. **No Error Isolation**: Well-designed plugin systems (like pytest hooks, Django middleware) isolate plugin failures to prevent one plugin from affecting others. The current implementation has no try/except protection around handler calls, violating this fundamental principle.

4. **Silent Data Loss**: The same issue affects `on_success` handlers (lines 121-122) where a handler exception causes the function's computed result to be lost instead of returned.

5. **Documentation Contradiction**: While not explicitly documented, the Protocol definitions in `pydantic/plugin/__init__.py` show default implementations that simply return, suggesting handlers shouldn't disrupt normal flow. The current behavior contradicts this implicit contract.

## Relevant Context

The bug exists in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py` at lines 107-123 in the `build_wrapper` function. The function creates a wrapper that calls event handlers at different stages of validation:

- `on_enter`: Called before validation (line 108)
- `on_error`: Called when ValidationError occurs (lines 113-114)
- `on_exception`: Called for other exceptions (lines 117-118)
- `on_success`: Called after successful validation (lines 121-122)

Each of these handler invocations lacks exception protection, meaning any handler that raises an exception will:
- Prevent subsequent handlers from running
- Replace the original error/result with the handler's exception
- Break the expected validation flow

The plugin system is marked as "Experimental" in the documentation, but this level of fragility makes it unusable even for experimental features in production environments.

Documentation: https://docs.pydantic.dev/latest/concepts/plugins/
Source code: https://github.com/pydantic/pydantic/blob/main/pydantic/plugin/_schema_validator.py

## Proposed Fix

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
+                    pass  # Could optionally log the exception

             try:
                 result = func(*args, **kwargs)
             except ValidationError as error:
                 for on_error_handler in on_errors:
-                    on_error_handler(error)
+                    try:
+                        on_error_handler(error)
+                    except Exception:
+                        pass  # Could optionally log the exception
                 raise
             except Exception as exception:
                 for on_exception_handler in on_exceptions:
-                    on_exception_handler(exception)
+                    try:
+                        on_exception_handler(exception)
+                    except Exception:
+                        pass  # Could optionally log the exception
                 raise
             else:
                 for on_success_handler in on_successes:
-                    on_success_handler(result)
+                    try:
+                        on_success_handler(result)
+                    except Exception:
+                        pass  # Could optionally log the exception
                 return result

         return wrapper
```