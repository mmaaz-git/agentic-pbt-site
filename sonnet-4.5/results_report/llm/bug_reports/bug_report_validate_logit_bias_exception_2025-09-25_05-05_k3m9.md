# Bug Report: validate_logit_bias Exception Handling

**Target**: `llm.default_plugins.openai_models.SharedOptions.validate_logit_bias`
**Severity**: Medium
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `validate_logit_bias` method's exception handling swallows specific error messages and replaces them with a generic error, providing poor user feedback when values are out of range.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import SharedOptions
import pytest

@given(
    st.dictionaries(
        st.integers(min_value=0, max_value=100000).map(str),
        st.integers().filter(lambda x: x < -100 or x > 100),
        min_size=1
    )
)
def test_validate_logit_bias_error_messages(logit_bias_dict):
    options = SharedOptions()

    with pytest.raises(ValueError) as exc_info:
        options.validate_logit_bias(logit_bias_dict)

    error_msg = str(exc_info.value)
    assert "Value must be between -100 and 100" in error_msg or \
           "Invalid key-value pair" in error_msg
```

**Failing behavior**: When a value is out of range (e.g., 150), the specific error message "Value must be between -100 and 100" is raised but then caught and replaced with the generic "Invalid key-value pair in logit_bias dictionary".

## Reproducing the Bug

```python
from llm.default_plugins.openai_models import SharedOptions

options = SharedOptions()

try:
    options.validate_logit_bias({"100": 150})
except ValueError as e:
    print(f"Error: {e}")
```

**Expected output**: `ValueError: Value must be between -100 and 100`
**Actual output**: `ValueError: Invalid key-value pair in logit_bias dictionary`

## Why This Is A Bug

The code explicitly raises a specific error message on line 427 when a value is out of range, but the catch-all `except ValueError` on line 428 immediately catches this exception and replaces it with a generic message. This violates the principle of providing clear error messages to users and makes it harder to debug issues with `logit_bias` values.

Users need to know whether:
- Their key couldn't be converted to an integer
- Their value couldn't be converted to an integer
- Their value is out of the valid range [-100, 100]

The current implementation loses this distinction.

## Fix

```diff
     @field_validator("logit_bias")
     def validate_logit_bias(cls, logit_bias):
         if logit_bias is None:
             return None

         if isinstance(logit_bias, str):
             try:
                 logit_bias = json.loads(logit_bias)
             except json.JSONDecodeError:
                 raise ValueError("Invalid JSON in logit_bias string")

         validated_logit_bias = {}
         for key, value in logit_bias.items():
             try:
                 int_key = int(key)
                 int_value = int(value)
-                if -100 <= int_value <= 100:
-                    validated_logit_bias[int_key] = int_value
-                else:
-                    raise ValueError("Value must be between -100 and 100")
-            except ValueError:
-                raise ValueError("Invalid key-value pair in logit_bias dictionary")
+            except ValueError as e:
+                raise ValueError(f"Invalid key-value pair in logit_bias dictionary: {key}={value}")
+
+            if not (-100 <= int_value <= 100):
+                raise ValueError(f"Value must be between -100 and 100, got {int_value}")
+
+            validated_logit_bias[int_key] = int_value

         return validated_logit_bias
```