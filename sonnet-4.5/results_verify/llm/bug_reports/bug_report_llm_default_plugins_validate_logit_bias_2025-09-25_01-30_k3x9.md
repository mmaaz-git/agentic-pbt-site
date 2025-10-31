# Bug Report: llm.default_plugins.openai_models validate_logit_bias None Handling

**Target**: `llm.default_plugins.openai_models.SharedOptions.validate_logit_bias`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `validate_logit_bias` validator crashes with a TypeError when processing dictionaries containing None values, instead of either filtering them out or raising a descriptive error message.

## Property-Based Test

```python
from hypothesis import given, strategies as st, assume
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Union
import json


class SharedOptions(BaseModel):
    logit_bias: Optional[Union[dict, str]] = Field(default=None)

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
                if -100 <= int_value <= 100:
                    validated_logit_bias[int_key] = int_value
                else:
                    raise ValueError("Value must be between -100 and 100")
            except ValueError:
                raise ValueError("Invalid key-value pair in logit_bias dictionary")

        return validated_logit_bias


@given(
    st.dictionaries(
        st.text(min_size=1, max_size=20),
        st.one_of(st.text(min_size=1), st.integers(), st.booleans(), st.none()),
        min_size=0,
        max_size=5
    )
)
def test_validate_logit_bias_rejects_invalid_keys(invalid_dict):
    assume(any(not k.lstrip('-').isdigit() for k in invalid_dict.keys()))

    try:
        options = SharedOptions(logit_bias=invalid_dict)
        assert False, f"Should have raised ValueError for {invalid_dict}"
    except ValueError:
        pass
```

**Failing input**: `{'0': None, ':': None}`

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import SharedOptions

options = SharedOptions(logit_bias={"1712": None})
```

This raises:
```
TypeError: int() argument must be a string, a bytes-like object or a real number, not 'NoneType'
```

## Why This Is A Bug

The `validate_logit_bias` method attempts to convert dictionary values to integers using `int(value)` without first checking if the value is None. This causes a TypeError instead of a clear validation error.

The function should either:
1. Skip/filter out None values silently, or
2. Raise a descriptive ValueError explaining that None values are not allowed in logit_bias

The current behavior provides a confusing error message that doesn't help users understand what went wrong.

## Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -419,6 +419,8 @@ class SharedOptions(llm.Options):
         validated_logit_bias = {}
         for key, value in logit_bias.items():
             try:
+                if value is None:
+                    continue
                 int_key = int(key)
                 int_value = int(value)
                 if -100 <= int_value <= 100:
```

Alternatively, to be more strict:

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -419,6 +419,8 @@ class SharedOptions(llm.Options):
         validated_logit_bias = {}
         for key, value in logit_bias.items():
             try:
+                if value is None:
+                    raise ValueError("logit_bias values cannot be None")
                 int_key = int(key)
                 int_value = int(value)
                 if -100 <= int_value <= 100:
```