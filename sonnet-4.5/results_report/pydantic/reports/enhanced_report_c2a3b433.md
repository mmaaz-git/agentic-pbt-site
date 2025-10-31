# Bug Report: pydantic.plugin._schema_validator ValueError Not Caught in Plugin Error Handling

**Target**: `pydantic.plugin._schema_validator.PluggableSchemaValidator.__init__`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When a plugin's `new_schema_validator` method returns a tuple with the wrong number of elements (not 3), a `ValueError` is raised during tuple unpacking but is not caught by the error handling code, which only catches `TypeError`. This results in an unhelpful generic error message that doesn't identify which plugin caused the problem.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from hypothesis import given, strategies as st, settings, assume
from pydantic.plugin._schema_validator import PluggableSchemaValidator
from pydantic.plugin import SchemaTypePath


@given(st.integers(min_value=0, max_value=10))
@settings(max_examples=50)
def test_plugin_wrong_tuple_size_error_message(tuple_size):
    assume(tuple_size != 3)

    class BadPlugin:
        def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
            return tuple([None] * tuple_size)

    schema = {'type': 'int'}
    schema_type = int
    schema_type_path = SchemaTypePath('test', 'test')
    schema_kind = 'TypeAdapter'

    try:
        validator = PluggableSchemaValidator(
            schema, schema_type, schema_type_path, schema_kind, None, [BadPlugin()], {}
        )
        raise AssertionError(f"No exception raised for tuple size {tuple_size}")
    except (TypeError, ValueError) as e:
        error_msg = str(e)
        has_plugin_info = ("BadPlugin" in error_msg or
                          "Error using plugin" in error_msg or
                          "test_valueerror_bug" in error_msg)
        if not has_plugin_info:
            raise AssertionError(
                f"BUG: Exception doesn't identify which plugin failed!\n"
                f"Tuple size: {tuple_size}\n"
                f"Exception type: {type(e).__name__}\n"
                f"Message: {error_msg}"
            )


if __name__ == "__main__":
    test_plugin_wrong_tuple_size_error_message()
```

<details>

<summary>
**Failing input**: `tuple_size=0`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 24, in test_plugin_wrong_tuple_size_error_message
    validator = PluggableSchemaValidator(
        schema, schema_type, schema_type_path, schema_kind, None, [BadPlugin()], {}
    )
  File "/home/npc/miniconda/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py", line 75, in __init__
    p, j, s = plugin.new_schema_validator(
    ^^^^^^^
ValueError: not enough values to unpack (expected 3, got 0)

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 43, in <module>
    test_plugin_wrong_tuple_size_error_message()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 10, in test_plugin_wrong_tuple_size_error_message
    @settings(max_examples=50)
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/44/hypo.py", line 34, in test_plugin_wrong_tuple_size_error_message
    raise AssertionError(
    ...<4 lines>...
    )
AssertionError: BUG: Exception doesn't identify which plugin failed!
Tuple size: 0
Exception type: ValueError
Message: not enough values to unpack (expected 3, got 0)
Falsifying example: test_plugin_wrong_tuple_size_error_message(
    tuple_size=0,
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from pydantic.plugin._schema_validator import PluggableSchemaValidator
from pydantic.plugin import SchemaTypePath


class BadPlugin:
    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        # Return wrong number of elements (2 instead of 3)
        return (None, None)


schema = {'type': 'int'}
schema_type = int
schema_type_path = SchemaTypePath('test', 'test')
schema_kind = 'TypeAdapter'

try:
    validator = PluggableSchemaValidator(
        schema, schema_type, schema_type_path, schema_kind, None, [BadPlugin()], {}
    )
    print("ERROR: No exception was raised!")
except ValueError as e:
    print(f"ValueError: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Does error identify plugin? {'BadPlugin' in str(e) or 'Error using plugin' in str(e)}")
except TypeError as e:
    print(f"TypeError: {e}")
    print(f"Error type: {type(e).__name__}")
    print(f"Does error identify plugin? {'BadPlugin' in str(e) or 'Error using plugin' in str(e)}")
```

<details>

<summary>
ValueError raised without plugin identification
</summary>
```
ValueError: not enough values to unpack (expected 3, got 2)
Error type: ValueError
Does error identify plugin? False
```
</details>

## Why This Is A Bug

This violates the expected behavior and API contract for several reasons:

1. **Documentation Contract**: The `PydanticPluginProtocol` at line 25 of `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/plugin/__init__.py` explicitly defines that `new_schema_validator` must return a 3-tuple via the type alias: `NewSchemaReturns: TypeAlias = 'tuple[ValidatePythonHandlerProtocol | None, ValidateJsonHandlerProtocol | None, ValidateStringsHandlerProtocol | None]'`

2. **Error Handling Intent**: The code at lines 79-80 of `_schema_validator.py` shows clear intent to catch plugin errors and enhance them with plugin identification:
   ```python
   except TypeError as e:  # pragma: no cover
       raise TypeError(f'Error using plugin `{plugin.__module__}:{plugin.__class__.__name__}`: {e}') from e
   ```
   This demonstrates that the developers intended all plugin errors to be caught and enhanced with context about which plugin failed.

3. **Python Behavior Mismatch**: The tuple unpacking operation `p, j, s = plugin.new_schema_validator(...)` at lines 76-77 raises `ValueError` (not `TypeError`) when the tuple has the wrong number of elements:
   - Too few elements: `ValueError: not enough values to unpack (expected 3, got N)`
   - Too many elements: `ValueError: too many values to unpack (expected 3)`

4. **Developer Experience Impact**: When multiple plugins are registered, the generic ValueError message makes it extremely difficult to identify which plugin is misconfigured, defeating the purpose of the error handling code that exists specifically to provide this context.

## Relevant Context

The bug occurs in the `PluggableSchemaValidator.__init__` method when processing plugins. The error handling code was clearly designed to provide helpful error messages to plugin developers, as evidenced by the formatted error message that includes the plugin's module and class name. However, the implementation only catches `TypeError`, missing the common case of `ValueError` from tuple unpacking.

This is particularly problematic because:
- Plugin development often involves trial and error
- Multiple plugins may be registered simultaneously
- The generic ValueError message provides no hint about which plugin is misconfigured
- The existing error handling pattern shows this was an oversight, not intentional

Links:
- Source code location: `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/plugin/_schema_validator.py:75-80`
- Protocol definition: `/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages/pydantic/plugin/__init__.py:25,41-51`

## Proposed Fix

```diff
--- a/pydantic/plugin/_schema_validator.py
+++ b/pydantic/plugin/_schema_validator.py
@@ -76,8 +76,8 @@ class PluggableSchemaValidator:
                 p, j, s = plugin.new_schema_validator(
                     schema, schema_type, schema_type_path, schema_kind, config, plugin_settings
                 )
-            except TypeError as e:  # pragma: no cover
-                raise TypeError(f'Error using plugin `{plugin.__module__}:{plugin.__class__.__name__}`: {e}') from e
+            except (TypeError, ValueError) as e:  # pragma: no cover
+                raise type(e)(f'Error using plugin `{plugin.__module__}:{plugin.__class__.__name__}`: {e}') from e
             if p is not None:
                 python_event_handlers.append(p)
             if j is not None:
```