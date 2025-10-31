# Bug Report: llm.default_plugins.openai_models.SharedOptions.validate_logit_bias Error Handling

**Target**: `llm.default_plugins.openai_models.SharedOptions.validate_logit_bias`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `validate_logit_bias()` field validator catches and masks specific error messages about out-of-range values, providing only a generic error message to users.

## Property-Based Test

```python
from hypothesis import given, strategies as st
import json

def validate_logit_bias(logit_bias):
    if logit_bias is None:
        return None

    if isinstance(logit_bias, str):
        logit_bias = json.loads(logit_bias)

    validated_logit_bias = {}
    for key, value in logit_bias.items():
        try:
            int_key = int(key)
            int_value = int(value)
            if -100 <= int_value <= 100:
                validated_logit_bias[int_key] = int_value
            else:
                raise ValueError("Value must be between -100 and 100")
        except ValueError:
            raise ValueError("Invalid key-value pair in logit_bias dictionary")

    return validated_logit_bias

@given(st.dictionaries(st.text(min_size=1), st.integers().filter(lambda x: x < -100 or x > 100), min_size=1))
def test_error_message_clarity(logit_bias):
    try:
        validate_logit_bias(logit_bias)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "between -100 and 100" in str(e), f"Expected specific error about range, got: {e}"
```

**Failing input**: Any dict with values outside [-100, 100], e.g., `{"100": 150}`

## Reproducing the Bug

```python
import json

def validate_logit_bias(logit_bias):
    if logit_bias is None:
        return None

    if isinstance(logit_bias, str):
        logit_bias = json.loads(logit_bias)

    validated_logit_bias = {}
    for key, value in logit_bias.items():
        try:
            int_key = int(key)
            int_value = int(value)
            if -100 <= int_value <= 100:
                validated_logit_bias[int_key] = int_value
            else:
                raise ValueError("Value must be between -100 and 100")
        except ValueError:
            raise ValueError("Invalid key-value pair in logit_bias dictionary")

    return validated_logit_bias

try:
    validate_logit_bias({"100": 150})
except ValueError as e:
    print(e)
```

**Output**: `Invalid key-value pair in logit_bias dictionary`
**Expected**: `Value must be between -100 and 100`

## Why This Is A Bug

When a user provides a value outside the valid range [-100, 100], the code raises a specific error message "Value must be between -100 and 100" on line 427. However, this ValueError is immediately caught by the except block on line 429, which re-raises it with the generic message "Invalid key-value pair in logit_bias dictionary".

This makes debugging difficult because users receive a vague error instead of being told the specific constraint violation. The error handling conflates two distinct error cases:
1. Invalid conversion to int (e.g., `{"a": "not_a_number"}`)
2. Valid int but out of range (e.g., `{"100": 150}`)

## Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -419,14 +419,15 @@ class SharedOptions(llm.Options):

         validated_logit_bias = {}
         for key, value in logit_bias.items():
             try:
                 int_key = int(key)
                 int_value = int(value)
-                if -100 <= int_value <= 100:
-                    validated_logit_bias[int_key] = int_value
-                else:
-                    raise ValueError("Value must be between -100 and 100")
             except ValueError:
                 raise ValueError("Invalid key-value pair in logit_bias dictionary")
+
+            if -100 <= int_value <= 100:
+                validated_logit_bias[int_key] = int_value
+            else:
+                raise ValueError("Value must be between -100 and 100")

         return validated_logit_bias
```