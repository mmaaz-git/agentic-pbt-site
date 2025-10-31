# Bug Report: pydantic.plugin._schema_validator ValueError Not Caught

**Target**: `pydantic.plugin._schema_validator.PluggableSchemaValidator.__init__`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

When a plugin's `new_schema_validator` method returns a tuple with the wrong number of elements, a `ValueError` is raised but not caught by the existing `except TypeError` clause. This results in an unhelpful error message that doesn't identify which plugin caused the problem.

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
```

**Failing input**: `tuple_size=0` (or any value != 3)

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env')

from pydantic.plugin._schema_validator import PluggableSchemaValidator
from pydantic.plugin import SchemaTypePath


class BadPlugin:
    def new_schema_validator(self, schema, schema_type, schema_type_path, schema_kind, config, plugin_settings):
        return (None, None)


schema = {'type': 'int'}
schema_type = int
schema_type_path = SchemaTypePath('test', 'test')
schema_kind = 'TypeAdapter'

try:
    validator = PluggableSchemaValidator(
        schema, schema_type, schema_type_path, schema_kind, None, [BadPlugin()], {}
    )
except ValueError as e:
    print(f"ValueError: {e}")
```

**Output:**
```
ValueError: not enough values to unpack (expected 3, got 2)
```

The error doesn't identify which plugin caused the problem, making debugging difficult for plugin authors.

## Why This Is A Bug

The code at `pydantic/plugin/_schema_validator.py:79-80` only catches `TypeError`:

```python
except TypeError as e:  # pragma: no cover
    raise TypeError(f'Error using plugin `{plugin.__module__}:{plugin.__class__.__name__}`: {e}') from e
```

However, tuple unpacking errors (`p, j, s = plugin.new_schema_validator(...)` at line 75-77) raise `ValueError`, not `TypeError`:
- Too few elements: `ValueError: not enough values to unpack (expected 3, got N)`
- Too many elements: `ValueError: too many values to unpack (expected 3)`

This means the error message doesn't include the helpful context about which plugin failed, violating the API contract that plugin errors should be clearly identified.

## Fix

```diff
--- a/pydantic/plugin/_schema_validator.py
+++ b/pydantic/plugin/_schema_validator.py
@@ -76,7 +76,7 @@ class PluggableSchemaValidator:
                 p, j, s = plugin.new_schema_validator(
                     schema, schema_type, schema_type_path, schema_kind, config, plugin_settings
                 )
-            except TypeError as e:  # pragma: no cover
-                raise TypeError(f'Error using plugin `{plugin.__module__}:{plugin.__class__.__name__}`: {e}') from e
+            except (TypeError, ValueError) as e:  # pragma: no cover
+                raise type(e)(f'Error using plugin `{plugin.__module__}:{plugin.__class__.__name__}`: {e}') from e
             if p is not None:
                 python_event_handlers.append(p)
```