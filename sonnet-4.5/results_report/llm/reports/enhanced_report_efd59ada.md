# Bug Report: llm.default_plugins.openai_models.not_nulls Dictionary Iteration Error

**Target**: `llm.default_plugins.openai_models.not_nulls`
**Severity**: High
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

The `not_nulls()` function crashes with a ValueError when called with a dictionary because it attempts to iterate directly over dictionary keys instead of key-value pairs.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import not_nulls

@given(st.dictionaries(st.text(), st.one_of(st.none(), st.integers(), st.text())))
def test_not_nulls_removes_none_values(data):
    result = not_nulls(data)
    for key, value in result.items():
        assert value is not None

if __name__ == "__main__":
    test_not_nulls_removes_none_values()
```

<details>

<summary>
**Failing input**: `{'': None}`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 14, in <module>
    test_not_nulls_removes_none_values()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 8, in test_not_nulls_removes_none_values
    def test_not_nulls_removes_none_values(data):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/26/hypo.py", line 9, in test_not_nulls_removes_none_values
    result = not_nulls(data)
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py", line 916, in not_nulls
    return {key: value for key, value in data if value is not None}
                           ^^^^^^^^^^
ValueError: not enough values to unpack (expected 2, got 0)
Falsifying example: test_not_nulls_removes_none_values(
    data={'': None},
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import not_nulls

# Test case that should work but crashes
test_dict = {'a': 1, 'b': None, 'c': 'test'}
print(f"Input: {test_dict}")
try:
    result = not_nulls(test_dict)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
```

<details>

<summary>
ValueError when iterating over dictionary
</summary>
```
Input: {'a': 1, 'b': None, 'c': 'test'}
Error: ValueError: not enough values to unpack (expected 2, got 1)
```
</details>

## Why This Is A Bug

The `not_nulls()` function is designed to filter out None values from a dictionary, but it contains a critical implementation error. When iterating over a dictionary with `for key, value in data`, Python iterates over just the keys, not key-value pairs. Each key is a single value (typically a string), and Python attempts to unpack it into two variables, causing a ValueError.

The function is called in `build_kwargs()` at line 658 with `prompt.options`, which is guaranteed to be a dictionary by the Prompt class constructor (line 365 in models.py: `self.options = options or {}`). This makes the entire OpenAI plugin unusable whenever options are provided, as `build_kwargs()` is called for all model executions (chat, async chat, and completion).

The bug only doesn't manifest with empty dictionaries because the iteration never executes. With any non-empty dictionary, even `{'': None}`, the unpacking error occurs immediately.

## Relevant Context

The `not_nulls` function is defined at line 916 of `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py`:

```python
def not_nulls(data) -> dict:
    return {key: value for key, value in data if value is not None}
```

It's used in the `build_kwargs` method (line 658) of the OpenAI model classes to process prompt options before sending them to the OpenAI API. The result is wrapped in `dict()`, confirming that dictionary-like output is expected.

This bug affects all OpenAI model implementations in the plugin (Chat, AsyncChat, and Completion classes) since they all use the same `build_kwargs` method. Users cannot pass any configuration options (like `max_tokens`, `temperature`, etc.) to OpenAI models without encountering this crash.

## Proposed Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -913,7 +913,7 @@ class Completion(Chat):


 def not_nulls(data) -> dict:
-    return {key: value for key, value in data if value is not None}
+    return {key: value for key, value in data.items() if value is not None}
```