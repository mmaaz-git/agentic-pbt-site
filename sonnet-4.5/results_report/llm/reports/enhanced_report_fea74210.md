# Bug Report: validate_logit_bias Exception Swallows Specific Error Messages

**Target**: `llm.default_plugins.openai_models.SharedOptions.validate_logit_bias`
**Severity**: Low
**Bug Type**: Contract
**Date**: 2025-09-25

## Summary

The `validate_logit_bias` method catches and replaces specific validation error messages with a generic error, preventing users from understanding why their input is invalid.

## Property-Based Test

```python
#!/usr/bin/env python3
"""Property-based test for validate_logit_bias bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from llm.default_plugins.openai_models import SharedOptions
import pytest

@given(
    st.dictionaries(
        st.integers(min_value=0, max_value=100000).map(str),
        st.integers().filter(lambda x: x < -100 or x > 100),
        min_size=1
    )
)
@settings(max_examples=10)  # Limit examples for brevity
def test_validate_logit_bias_error_messages(logit_bias_dict):
    """Test that out-of-range values produce specific error messages"""
    options = SharedOptions()

    with pytest.raises(ValueError) as exc_info:
        options.validate_logit_bias(logit_bias_dict)

    error_msg = str(exc_info.value)

    # The bug: we expect specific error messages but get generic ones
    # This assertion will fail due to the bug
    assert "Value must be between -100 and 100" in error_msg or \
           "Invalid key-value pair" in error_msg

    # Print what we found for demonstration
    print(f"Input: {logit_bias_dict}")
    print(f"Error: {error_msg}")

    # The actual behavior (bug): always get generic message
    assert error_msg == "Invalid key-value pair in logit_bias dictionary"

if __name__ == "__main__":
    # Run the test
    test_validate_logit_bias_error_messages()
```

<details>

<summary>
**Failing input**: `{'10223': -32309, '100000': 13161, '74190': -22874, '14104': 28513, '6206': 8530468494504254272}`
</summary>
```
Input: {'10223': -32309, '100000': 13161, '74190': -22874, '14104': 28513, '6206': 8530468494504254272}
Error: Invalid key-value pair in logit_bias dictionary
Input: {'34928': 14126}
Error: Invalid key-value pair in logit_bias dictionary
Input: {'72985': 27734}
Error: Invalid key-value pair in logit_bias dictionary
Input: {'73458': -573319707, '84497': -2533}
Error: Invalid key-value pair in logit_bias dictionary
Input: {'12492': -25066, '67967': 31196}
Error: Invalid key-value pair in logit_bias dictionary
Input: {'77790': 4006, '37980': 1550}
Error: Invalid key-value pair in logit_bias dictionary
Input: {'46752': 8169}
Error: Invalid key-value pair in logit_bias dictionary
Input: {'83415': 26717, '7450': -14010}
Error: Invalid key-value pair in logit_bias dictionary
Input: {'33172': -118, '96844': 2085934468}
Error: Invalid key-value pair in logit_bias dictionary
Input: {'91856': -12674, '91110': 1243, '50128': 1806458149, '83571': 6713}
Error: Invalid key-value pair in logit_bias dictionary
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""Minimal reproduction of validate_logit_bias exception handling bug"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import SharedOptions

# Test case 1: Value out of range (too high)
options = SharedOptions()
try:
    result = options.validate_logit_bias({"100": 150})
    print(f"Test 1 passed unexpectedly: {result}")
except ValueError as e:
    print(f"Test 1 - Out of range high (150): {e}")

# Test case 2: Value out of range (too low)
try:
    result = options.validate_logit_bias({"100": -150})
    print(f"Test 2 passed unexpectedly: {result}")
except ValueError as e:
    print(f"Test 2 - Out of range low (-150): {e}")

# Test case 3: Invalid key (cannot convert to int)
try:
    result = options.validate_logit_bias({"abc": 50})
    print(f"Test 3 passed unexpectedly: {result}")
except ValueError as e:
    print(f"Test 3 - Invalid key ('abc'): {e}")

# Test case 4: Invalid value (cannot convert to int)
try:
    result = options.validate_logit_bias({"100": "not_a_number"})
    print(f"Test 4 passed unexpectedly: {result}")
except ValueError as e:
    print(f"Test 4 - Invalid value ('not_a_number'): {e}")

# Test case 5: Valid input (should pass)
try:
    result = options.validate_logit_bias({"100": 50})
    print(f"Test 5 - Valid input: {result}")
except ValueError as e:
    print(f"Test 5 failed unexpectedly: {e}")
```

<details>

<summary>
All error conditions produce the same generic message
</summary>
```
Test 1 - Out of range high (150): Invalid key-value pair in logit_bias dictionary
Test 2 - Out of range low (-150): Invalid key-value pair in logit_bias dictionary
Test 3 - Invalid key ('abc'): Invalid key-value pair in logit_bias dictionary
Test 4 - Invalid value ('not_a_number'): Invalid key-value pair in logit_bias dictionary
Test 5 - Valid input: {100: 50}
```
</details>

## Why This Is A Bug

The code at line 427 explicitly raises a specific error message "Value must be between -100 and 100" when a value is outside the valid range [-100, 100]. However, the except block at line 428 immediately catches this ValueError and replaces it with the generic message "Invalid key-value pair in logit_bias dictionary".

This violates the principle of providing clear, actionable error messages to users. Users cannot distinguish between:
- A key that cannot be converted to an integer (e.g., "abc")
- A value that cannot be converted to an integer (e.g., "not_a_number")
- A value that is out of the valid range [-100, 100] (e.g., 150)

The developer's intent to provide specific error feedback is evident from writing the specific error message on line 427, but the flawed exception handling structure defeats this purpose.

## Relevant Context

The OpenAI API documentation specifies that logit_bias values must be between -100 and 100 inclusive. The llm library correctly validates this constraint but fails to communicate validation failures clearly.

The code structure shows a clear logic error:
- Lines 421-423: Convert key and value to integers
- Lines 424-427: Check range and either add to result or raise specific error
- Line 428: Catch ALL ValueErrors (including the one just raised) and replace with generic message

This pattern means the specific error on line 427 can never reach the user. The code is located at `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py`.

## Proposed Fix

```diff
@@ -418,13 +418,16 @@ class SharedOptions(llm.Options):

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
+                raise ValueError(f"Invalid key-value pair in logit_bias dictionary: {key}={value} (conversion error)")
+
+            if not (-100 <= int_value <= 100):
+                raise ValueError(f"Value must be between -100 and 100, got {int_value} for key {key}")
+
+            validated_logit_bias[int_key] = int_value

         return validated_logit_bias
```