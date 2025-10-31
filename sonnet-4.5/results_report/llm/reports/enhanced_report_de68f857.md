# Bug Report: llm.default_plugins.openai_models.validate_logit_bias Masks Specific Range Error Message

**Target**: `llm.default_plugins.openai_models.SharedOptions.validate_logit_bias`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `validate_logit_bias` validator suppresses its own specific error message "Value must be between -100 and 100" due to overly broad exception handling, showing users a generic "Invalid key-value pair" message instead.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

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
        assert "between -100 and 100" in error_msg, f"Expected 'between -100 and 100' in error message, but got: {error_msg}"

# Run the test
if __name__ == "__main__":
    test_logit_bias_out_of_range_error_message()
```

<details>

<summary>
**Failing input**: `{"123": 101}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 12, in test_logit_bias_out_of_range_error_message
    SharedOptions(logit_bias={"123": out_of_range_value})
    ~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/pydantic/main.py", line 253, in __init__
    validated_self = self.__pydantic_validator__.validate_python(data, self_instance=self)
pydantic_core._pydantic_core.ValidationError: 1 validation error for SharedOptions
logit_bias
  Value error, Invalid key-value pair in logit_bias dictionary [type=value_error, input_value={'123': 101}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/value_error

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 21, in <module>
    test_logit_bias_out_of_range_error_message()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 9, in test_logit_bias_out_of_range_error_message
    def test_logit_bias_out_of_range_error_message(out_of_range_value):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/3/hypo.py", line 17, in test_logit_bias_out_of_range_error_message
    assert "between -100 and 100" in error_msg, f"Expected 'between -100 and 100' in error message, but got: {error_msg}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 'between -100 and 100' in error message, but got: 1 validation error for SharedOptions
logit_bias
  Value error, Invalid key-value pair in logit_bias dictionary [type=value_error, input_value={'123': 101}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/value_error
Falsifying example: test_logit_bias_out_of_range_error_message(
    out_of_range_value=101,  # or any other generated value
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import SharedOptions
from pydantic import ValidationError

try:
    # Test with value above the allowed range
    opts = SharedOptions(logit_bias={"123": 150})
except ValidationError as e:
    print(f"Error for value 150: {e}")

print("\n" + "="*50 + "\n")

try:
    # Test with value below the allowed range
    opts = SharedOptions(logit_bias={"456": -150})
except ValidationError as e:
    print(f"Error for value -150: {e}")

print("\n" + "="*50 + "\n")

try:
    # Test with value at upper boundary (should work)
    opts = SharedOptions(logit_bias={"789": 100})
    print("Value 100: SUCCESS - No error raised")
except ValidationError as e:
    print(f"Error for value 100: {e}")

print("\n" + "="*50 + "\n")

try:
    # Test with value at lower boundary (should work)
    opts = SharedOptions(logit_bias={"101": -100})
    print("Value -100: SUCCESS - No error raised")
except ValidationError as e:
    print(f"Error for value -100: {e}")

print("\n" + "="*50 + "\n")

try:
    # Test with invalid key (non-numeric)
    opts = SharedOptions(logit_bias={"abc": 50})
except ValidationError as e:
    print(f"Error for non-numeric key: {e}")
```

<details>

<summary>
Values outside [-100, 100] range produce generic error instead of specific range error
</summary>
```
Error for value 150: 1 validation error for SharedOptions
logit_bias
  Value error, Invalid key-value pair in logit_bias dictionary [type=value_error, input_value={'123': 150}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/value_error

==================================================

Error for value -150: 1 validation error for SharedOptions
logit_bias
  Value error, Invalid key-value pair in logit_bias dictionary [type=value_error, input_value={'456': -150}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/value_error

==================================================

Value 100: SUCCESS - No error raised

==================================================

Value -100: SUCCESS - No error raised

==================================================

Error for non-numeric key: 1 validation error for SharedOptions
logit_bias
  Value error, Invalid key-value pair in logit_bias dictionary [type=value_error, input_value={'abc': 50}, input_type=dict]
    For further information visit https://errors.pydantic.dev/2.11/v/value_error
```
</details>

## Why This Is A Bug

This violates the principle of providing helpful error messages. The code explicitly attempts to provide a specific error message at line 427 (`raise ValueError("Value must be between -100 and 100")`), but this message is immediately caught by the overly broad `except ValueError` clause at line 428 and replaced with the generic "Invalid key-value pair in logit_bias dictionary" message.

The OpenAI API documentation specifies that logit_bias values must be between -100 and 100, where -100 effectively blocks a token and 100 makes it exclusively likely. Users encountering the generic error have no indication of what's wrong with their input - they don't know if the issue is with the key format, value format, or value range. The specific error message already exists in the code but never reaches users due to this exception handling anti-pattern.

## Relevant Context

The `logit_bias` parameter is part of the OpenAI API specification for controlling token probabilities during text generation. According to OpenAI's documentation:
- Values must be integers between -100 and 100
- Token IDs are specified as string keys in JSON
- -100 typically blocks a token completely
- 100 makes a token exclusively likely to be selected

The code at `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py` lines 420-431 shows the problematic exception handling pattern where the same `ValueError` type is used for both conversion failures and range validation, making it impossible for the catch-all handler to distinguish between them.

## Proposed Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -419,13 +419,14 @@ class SharedOptions(llm.Options):
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
+
+            validated_logit_bias[int_key] = int_value

         return validated_logit_bias
```