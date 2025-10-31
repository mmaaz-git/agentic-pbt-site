# Bug Report: llm.default_plugins.openai_models validate_logit_bias Error Message

**Target**: `llm.default_plugins.openai_models.SharedOptions.validate_logit_bias`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `validate_logit_bias` validator has a catch-all `except ValueError` that shadows the specific error message "Value must be between -100 and 100". When a value is out of range, users see the generic "Invalid key-value pair in logit_bias dictionary" instead of the helpful range error.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import SharedOptions
from pydantic import ValidationError

@given(st.integers(min_value=101))
def test_logit_bias_out_of_range_error_message(out_of_range_value):
    """Property: out-of-range values should give specific error message"""
    try:
        SharedOptions(logit_bias={"123": out_of_range_value})
        assert False, "Should have raised ValidationError"
    except ValidationError as e:
        error_msg = str(e)
        # Should see the specific range error, not generic message
        assert "between -100 and 100" in error_msg
```

**Failing input**: `{"123": 150}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import SharedOptions
from pydantic import ValidationError

try:
    opts = SharedOptions(logit_bias={"123": 150})
except ValidationError as e:
    print(f"Error: {e}")
```

**Output**:
```
Error: 1 validation error for SharedOptions
logit_bias
  Value error, Invalid key-value pair in logit_bias dictionary [type=value_error, ...]
```

**Expected**:
```
Error: 1 validation error for SharedOptions
logit_bias
  Value error, Value must be between -100 and 100 [type=value_error, ...]
```

## Why This Is A Bug

The validator raises a specific `ValueError("Value must be between -100 and 100")` on line 427, but this is immediately caught by the generic `except ValueError` on line 428, which re-raises with a less helpful message. This violates the principle of informative error messages and makes debugging harder for users.

The issue is that the same `ValueError` exception is used for two different cases:
1. Value conversion failure (line 422-423)
2. Range validation failure (line 427)

The catch-all handler cannot distinguish between these cases.

## Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -419,13 +419,13 @@ class SharedOptions(llm.Options):
         validated_logit_bias = {}
         for key, value in logit_bias.items():
             try:
                 int_key = int(key)
-                int_value = int(value)
-                if -100 <= int_value <= 100:
-                    validated_logit_bias[int_key] = int_value
-                else:
-                    raise ValueError("Value must be between -100 and 100")
-            except ValueError:
+            except (ValueError, TypeError):
                 raise ValueError("Invalid key-value pair in logit_bias dictionary")
+
+            int_value = int(value)
+            if not (-100 <= int_value <= 100):
+                raise ValueError("Value must be between -100 and 100")
+            validated_logit_bias[int_key] = int_value

         return validated_logit_bias
```