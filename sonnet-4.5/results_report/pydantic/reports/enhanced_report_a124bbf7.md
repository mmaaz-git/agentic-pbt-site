# Bug Report: pydantic.plugin._schema_validator.build_wrapper on_error Handler Exception Suppresses Original ValidationError

**Target**: `pydantic.plugin._schema_validator.build_wrapper`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a plugin's `on_error` handler raises an exception while processing a `ValidationError`, the handler's exception completely replaces the original `ValidationError`, violating the expected behavior that validation errors should always be propagated to the caller.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

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

if __name__ == "__main__":
    test_on_error_handler_exception_suppresses_original()
```

<details>

<summary>
**Failing input**: `original_error_msg='', handler_error_msg=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py", line 110, in wrapper
    result = func(*args, **kwargs)
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 14, in func
    raise ValidationError.from_exception_data(original_error_msg, [])
pydantic_core._pydantic_core.ValidationError: 0 validation errors for


During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 26, in test_on_error_handler_exception_suppresses_original
    wrapper()
    ~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py", line 113, in wrapper
    on_error_handler(error)
    ~~~~~~~~~~~~~~~~^^^^^^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 19, in bad_on_error
    raise RuntimeError(handler_error_msg)
RuntimeError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 34, in <module>
    test_on_error_handler_exception_suppresses_original()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 11, in test_on_error_handler_exception_suppresses_original
    @settings(max_examples=20)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/37/hypo.py", line 29, in test_on_error_handler_exception_suppresses_original
    raise AssertionError(f"Original ValidationError was suppressed by handler exception")
AssertionError: Original ValidationError was suppressed by handler exception
Falsifying example: test_on_error_handler_exception_suppresses_original(
    # The test always failed when commented parts were varied together.
    original_error_msg='',  # or any other generated value
    handler_error_msg='',  # or any other generated value
)
```
</details>

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

<details>

<summary>
BUG: RuntimeError replaces original ValidationError
</summary>
```
BUG: Caught RuntimeError from handler: Handler raised an exception
The original ValidationError was suppressed!
```
</details>

## Why This Is A Bug

This behavior violates several core expectations of the pydantic validation system:

1. **Violation of Handler Contract**: The `BaseValidateHandlerProtocol.on_error` method is documented as a "callback to be notified of validation errors" with return type `None`. The name and documentation strongly suggest handlers should be observational and not affect control flow. When a handler raises an exception, it breaks this contract by replacing the original error.

2. **Loss of Critical Error Information**: The original `ValidationError` contains crucial information about what validation failed and why. When replaced by a handler's exception, this information is completely lost, making it impossible to debug validation failures. The stack trace shows the handler error occurred "During handling of the above exception" but the original error is never re-raised.

3. **Breaks Exception Propagation Guarantee**: The code at lines 112-115 in `_schema_validator.py` clearly intends to re-raise the original `ValidationError` with the `raise` statement on line 115. However, if any handler raises an exception on line 114, execution never reaches the re-raise, breaking this guarantee.

4. **Plugin System Instability**: A single misbehaving plugin can corrupt the validation process for an entire application. Since plugins are meant to extend functionality without breaking core behavior, allowing them to suppress validation errors undermines system reliability.

5. **Inconsistent with Other Handler Types**: The `on_exception` handler (lines 116-119) properly re-raises the original exception after notifying handlers, maintaining consistency. The `on_error` handler should behave similarly.

## Relevant Context

The pydantic plugin system is marked as experimental in the documentation. The relevant code is in `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py` at lines 112-115:

```python
except ValidationError as error:
    for on_error_handler in on_errors:
        on_error_handler(error)  # If this raises, we never reach the next line
    raise  # This re-raise is skipped if handler raises
```

The `BaseValidateHandlerProtocol` in `pydantic/plugin/__init__.py` defines `on_error` with a docstring stating it's a "Callback to be notified of validation errors" and returns `None`, suggesting it should not affect the validation flow.

Documentation: https://docs.pydantic.dev/latest/concepts/plugins/

## Proposed Fix

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
+                            f'Exception in plugin on_error handler: {handler_exception!r}',
+                            RuntimeWarning
+                        )
                 raise
             except Exception as exception:
                 for on_exception_handler in on_exceptions:
```