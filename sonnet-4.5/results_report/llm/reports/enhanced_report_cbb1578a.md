# Bug Report: llm.default_plugins.openai_models combine_chunks Role Field Incorrectly Overwritten

**Target**: `llm.default_plugins.openai_models.combine_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `combine_chunks` function incorrectly overwrites the `role` field with `None` from subsequent chunks, losing the original `role="assistant"` from the first chunk when combining OpenAI streaming responses.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import combine_chunks

class MockDelta:
    def __init__(self, role=None, content=None):
        self.role = role
        self.content = content

class MockChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason
        self.logprobs = None

class MockChunk:
    def __init__(self, choices):
        self.choices = choices
        self.usage = None

@given(st.lists(st.text(), min_size=1))
def test_combine_chunks_preserves_role(content_pieces):
    """Property: role from first chunk should be preserved"""
    chunks = []
    for i, content in enumerate(content_pieces):
        role = "assistant" if i == 0 else None
        chunks.append(MockChunk([MockChoice(MockDelta(role=role, content=content))]))

    result = combine_chunks(chunks)
    assert result['role'] == "assistant", f"Expected 'assistant', got {result['role']}"

if __name__ == "__main__":
    test_combine_chunks_preserves_role()
```

<details>

<summary>
**Failing input**: `['', '']`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 35, in <module>
    test_combine_chunks_preserves_role()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 24, in test_combine_chunks_preserves_role
    def test_combine_chunks_preserves_role(content_pieces):
                   ^^^
  File "/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/0/hypo.py", line 32, in test_combine_chunks_preserves_role
    assert result['role'] == "assistant", f"Expected 'assistant', got {result['role']}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 'assistant', got None
Falsifying example: test_combine_chunks_preserves_role(
    content_pieces=['', ''],
)
```
</details>

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

from llm.default_plugins.openai_models import combine_chunks

class MockDelta:
    def __init__(self, role=None, content=None):
        self.role = role
        self.content = content

class MockChoice:
    def __init__(self, delta, finish_reason=None):
        self.delta = delta
        self.finish_reason = finish_reason
        self.logprobs = None

class MockChunk:
    def __init__(self, choices):
        self.choices = choices
        self.usage = None
        self.id = "test"
        self.object = "chat.completion.chunk"
        self.model = "gpt-4"
        self.created = 1234567890
        self.index = 0

# Test case: First chunk has role="assistant", subsequent chunks have role=None
chunks = [
    MockChunk([MockChoice(MockDelta(role="assistant", content="Hello"))]),
    MockChunk([MockChoice(MockDelta(role=None, content=" world"))]),
]

result = combine_chunks(chunks)
print(f"Content: {result['content']}")
print(f"Role: {result['role']}")
print()
print("Expected:")
print("Content: Hello world")
print("Role: assistant")
print()
print("Actual bug: The role is None instead of assistant")
```

<details>

<summary>
Role field incorrectly becomes None instead of preserving "assistant"
</summary>
```
Content: Hello world
Role: None

Expected:
Content: Hello world
Role: assistant

Actual bug: The role is None instead of assistant
```
</details>

## Why This Is A Bug

This bug violates the expected behavior of OpenAI's streaming API integration. In the OpenAI streaming protocol, the `role` field is deliberately sent only in the first chunk to identify the message source (e.g., "assistant"), while subsequent chunks have `role=None` to avoid redundancy. The `combine_chunks` function is supposed to aggregate these chunks into a single complete response, preserving all relevant metadata.

The bug occurs at line 943 in `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py` where `role = choice.delta.role` unconditionally overwrites the role variable on every chunk iteration. This means that when processing multiple chunks, the valid role from the first chunk gets replaced by `None` from all subsequent chunks, resulting in complete loss of role information in the combined response.

This directly contradicts the expected behavior of preserving metadata when combining streaming chunks, and breaks compatibility with code that depends on the role field to identify message sources in chat completions.

## Relevant Context

The `combine_chunks` function is used internally by the llm package in three locations within openai_models.py:
- Line 734: For chat completions with streaming
- Line 832: For async chat completions with streaming
- Line 900: For legacy completions with streaming

All usages wrap the result with `remove_dict_none_values()` which suggests that None values are expected to be cleaned up afterward, but the role field being None is a data loss issue, not just a cleanup concern.

The function lacks documentation about its expected behavior, but based on OpenAI's well-established streaming API patterns and the function's purpose of combining chunks without losing information, the current behavior is clearly incorrect.

Code location: `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py:919-963`

## Proposed Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -940,7 +940,8 @@ def combine_chunks(chunks: List) -> dict:
             if not hasattr(choice, "delta"):
                 content += choice.text
                 continue
-            role = choice.delta.role
+            if choice.delta.role is not None:
+                role = choice.delta.role
             if choice.delta.content is not None:
                 content += choice.delta.content
             if choice.finish_reason is not None:
```