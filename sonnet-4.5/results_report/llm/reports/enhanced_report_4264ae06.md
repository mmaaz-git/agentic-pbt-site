# Bug Report: llm.default_plugins.openai_models.combine_chunks Loses Role Information in Streaming Responses

**Target**: `llm.default_plugins.openai_models.combine_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `combine_chunks()` function incorrectly overwrites the role field with None when processing multi-chunk streaming responses, causing the assistant role metadata to be lost in standard OpenAI API streaming scenarios.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from typing import List, Optional
from dataclasses import dataclass

@dataclass
class Delta:
    role: Optional[str]
    content: Optional[str]

@dataclass
class Choice:
    delta: Delta
    finish_reason: Optional[str]

    @property
    def logprobs(self):
        return None

@dataclass
class Chunk:
    choices: List[Choice]
    usage: Optional[object] = None

def combine_chunks(chunks: List) -> dict:
    content = ""
    role = None
    finish_reason = None
    # If any of them have log probability, we're going to persist
    # those later on
    logprobs = []
    usage = {}

    for item in chunks:
        if item.usage:
            usage = item.usage.model_dump()
        for choice in item.choices:
            if choice.logprobs and hasattr(choice.logprobs, "top_logprobs"):
                logprobs.append(
                    {
                        "text": choice.text if hasattr(choice, "text") else None,
                        "top_logprobs": choice.logprobs.top_logprobs,
                    }
                )

            if not hasattr(choice, "delta"):
                content += choice.text
                continue
            role = choice.delta.role
            if choice.delta.content is not None:
                content += choice.delta.content
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    # Imitations of the OpenAI API may be missing some of these fields
    combined = {
        "content": content,
        "role": role,
        "finish_reason": finish_reason,
        "usage": usage,
    }
    if logprobs:
        combined["logprobs"] = logprobs
    if chunks:
        for key in ("id", "object", "model", "created", "index"):
            value = getattr(chunks[0], key, None)
            if value is not None:
                combined[key] = value

    return combined

@given(st.integers(min_value=2, max_value=10))
def test_combine_chunks_preserves_role(num_chunks):
    chunks = [
        Chunk(choices=[Choice(delta=Delta(role="assistant", content="Hello"), finish_reason=None)])
    ]
    for _ in range(num_chunks - 1):
        chunks.append(
            Chunk(choices=[Choice(delta=Delta(role=None, content=" more"), finish_reason=None)])
        )

    result = combine_chunks(chunks)
    assert result['role'] == "assistant", f"Expected 'assistant' but got {result['role']}"

if __name__ == "__main__":
    # Run the property-based test
    test_combine_chunks_preserves_role()
```

<details>

<summary>
**Failing input**: `num_chunks=2`
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 86, in <module>
    test_combine_chunks_preserves_role()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 72, in test_combine_chunks_preserves_role
    def test_combine_chunks_preserves_role(num_chunks):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/52/hypo.py", line 82, in test_combine_chunks_preserves_role
    assert result['role'] == "assistant", f"Expected 'assistant' but got {result['role']}"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: Expected 'assistant' but got None
Falsifying example: test_combine_chunks_preserves_role(
    num_chunks=2,
)
```
</details>

## Reproducing the Bug

```python
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class Delta:
    role: Optional[str]
    content: Optional[str]

@dataclass
class Choice:
    delta: Delta
    finish_reason: Optional[str]

    @property
    def logprobs(self):
        return None

@dataclass
class Chunk:
    choices: List[Choice]
    usage: Optional[object] = None

def combine_chunks(chunks: List) -> dict:
    content = ""
    role = None
    finish_reason = None
    # If any of them have log probability, we're going to persist
    # those later on
    logprobs = []
    usage = {}

    for item in chunks:
        if item.usage:
            usage = item.usage.model_dump()
        for choice in item.choices:
            if choice.logprobs and hasattr(choice.logprobs, "top_logprobs"):
                logprobs.append(
                    {
                        "text": choice.text if hasattr(choice, "text") else None,
                        "top_logprobs": choice.logprobs.top_logprobs,
                    }
                )

            if not hasattr(choice, "delta"):
                content += choice.text
                continue
            role = choice.delta.role
            if choice.delta.content is not None:
                content += choice.delta.content
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    # Imitations of the OpenAI API may be missing some of these fields
    combined = {
        "content": content,
        "role": role,
        "finish_reason": finish_reason,
        "usage": usage,
    }
    if logprobs:
        combined["logprobs"] = logprobs
    if chunks:
        for key in ("id", "object", "model", "created", "index"):
            value = getattr(chunks[0], key, None)
            if value is not None:
                combined[key] = value

    return combined

# Test case mimicking OpenAI streaming behavior
chunks = [
    # First chunk typically contains the role
    Chunk(choices=[Choice(delta=Delta(role="assistant", content="Hello"), finish_reason=None)]),
    # Subsequent chunks have role=None but contain content
    Chunk(choices=[Choice(delta=Delta(role=None, content=" world"), finish_reason=None)]),
]

result = combine_chunks(chunks)
print(f"Result role: {result['role']}")
print(f"Result content: {result['content']}")
print(f"Expected role: 'assistant'")
print(f"Bug: role is {result['role']} instead of 'assistant'")
```

<details>

<summary>
Role field is incorrectly set to None instead of "assistant"
</summary>
```
Result role: None
Result content: Hello world
Expected role: 'assistant'
Bug: role is None instead of 'assistant'
```
</details>

## Why This Is A Bug

This violates the expected behavior of OpenAI's streaming API response handling. According to OpenAI's API documentation, in streaming responses:

1. The first chunk contains the role field (typically "assistant") along with initial content
2. Subsequent chunks have `delta.role = None` and only contain additional content pieces
3. The combined response should preserve the role from the first chunk

The current implementation unconditionally assigns `role = choice.delta.role` on line 943 of `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py`. This means that when processing the second chunk (where `delta.role` is None), it overwrites the valid "assistant" role from the first chunk with None.

This breaks downstream functionality that relies on the role field for:
- Conversation history tracking
- Message formatting
- API response validation
- Debugging and logging

## Relevant Context

The bug occurs in the `combine_chunks` function which is called by both the synchronous `Chat.execute()` method (line 734) and asynchronous `AsyncChat.execute()` method (line 832) when processing streaming responses.

The function correctly accumulates content from all chunks but fails to preserve the role metadata. The same pattern of unconditional assignment also affects the `finish_reason` field, though that's less critical since it's typically only present in the final chunk.

OpenAI API streaming documentation: https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream
Source code location: `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py:919-964`

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