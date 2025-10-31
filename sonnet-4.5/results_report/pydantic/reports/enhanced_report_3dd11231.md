# Bug Report: pydantic.plugin._schema_validator.build_wrapper Exception Handler Suppresses Original Exceptions

**Target**: `pydantic.plugin._schema_validator.build_wrapper`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

When a plugin's `on_exception` handler raises an exception while processing a non-ValidationError exception, the handler's exception completely replaces and suppresses the original exception, making it impossible to debug the actual error in the application code.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

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


if __name__ == "__main__":
    test_on_exception_handler_exception_suppresses_original()
```

<details>

<summary>
**Failing input**: `original_error_msg='', handler_error_msg=''`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py", line 110, in wrapper
    result = func(*args, **kwargs)
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 13, in func
    raise ValueError(original_error_msg)
ValueError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 25, in test_on_exception_handler_exception_suppresses_original
    wrapper()
    ~~~~~~~^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py", line 117, in wrapper
    on_exception_handler(exception)
    ~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 18, in bad_on_exception
    raise RuntimeError(handler_error_msg)
RuntimeError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 34, in <module>
    test_on_exception_handler_exception_suppresses_original()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 10, in test_on_exception_handler_exception_suppresses_original
    @settings(max_examples=20)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 28, in test_on_exception_handler_exception_suppresses_original
    raise AssertionError(f"Original exception was suppressed by handler exception")
AssertionError: Original exception was suppressed by handler exception
Falsifying example: test_on_exception_handler_exception_suppresses_original(
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

<details>

<summary>
BUG: RuntimeError suppresses original ValueError
</summary>
```
BUG: Caught RuntimeError from handler: Handler raised an exception
The original ValueError was suppressed!
```
</details>

## Why This Is A Bug

This behavior violates fundamental exception handling principles and the expected contract of plugin handlers. Plugin handlers should be observational components that monitor the validation process without interfering with the normal exception flow. According to plugin architecture best practices, handlers should:

1. **Never suppress original exceptions**: The original application exception contains critical debugging information including the stack trace, error message, and context about where the error occurred in the actual business logic.

2. **Be transparent to the application flow**: Plugin handlers are meant to observe and react to events, not replace them. When a handler fails, it should not prevent the original exception from propagating.

3. **Maintain debugging capability**: By replacing the original exception with the handler's exception, developers lose the ability to debug the actual problem in their application code. Instead, they see an error from the plugin system which may be completely unrelated.

The current implementation at lines 116-119 in `/home/npc/miniconda/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py` directly calls `on_exception_handler(exception)` without any protection against handler failures. When the handler raises an exception, Python's normal exception handling causes the handler's exception to replace the original one.

## Relevant Context

The issue occurs in the `build_wrapper` function which creates wrapper functions for validation methods. This wrapper is responsible for calling plugin handlers at different stages of validation:
- `on_enter`: Called before validation
- `on_success`: Called after successful validation
- `on_error`: Called when a ValidationError occurs
- `on_exception`: Called when any other Exception occurs

The bug specifically affects the `on_exception` handler path (lines 116-119). There's a similar potential issue with `on_error` handlers (lines 113-114) that handle ValidationErrors.

Code location: `/home/npc/miniconda/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py:116-119`

This issue would affect any Pydantic application using plugins where:
- The plugin has an `on_exception` handler implemented
- The handler itself can raise exceptions (e.g., due to bugs, network issues, resource constraints)
- The application relies on catching and handling specific exception types

## Proposed Fix

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
+                            f'Exception in plugin on_error handler: {handler_exception!r}',
+                            RuntimeWarning
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
+                            f'Exception in plugin on_exception handler: {handler_exception!r}',
+                            RuntimeWarning
+                        )
                 raise
             else:
                 for on_success_handler in on_successes:
```