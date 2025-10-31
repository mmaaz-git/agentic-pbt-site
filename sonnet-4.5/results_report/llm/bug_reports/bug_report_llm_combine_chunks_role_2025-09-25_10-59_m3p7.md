# Bug Report: llm.default_plugins.openai_models combine_chunks Role Overwrite

**Target**: `llm.default_plugins.openai_models.combine_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `combine_chunks` function unconditionally overwrites the `role` field with each chunk's `delta.role`, even when it's `None`. In typical OpenAI streaming responses, only the first chunk has `role="assistant"` and subsequent chunks have `role=None`, causing the final combined result to have `role=None` instead of `role="assistant"`.

## Property-Based Test

```python
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
```

**Failing input**: Any list with more than one element, e.g., `["Hello", " world"]`

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

chunks = [
    MockChunk([MockChoice(MockDelta(role="assistant", content="Hello"))]),
    MockChunk([MockChoice(MockDelta(role=None, content=" world"))]),
]

result = combine_chunks(chunks)
print(f"Content: {result['content']}")
print(f"Role: {result['role']}")
```

**Output**:
```
Content: Hello world
Role: None
```

**Expected**:
```
Content: Hello world
Role: assistant
```

## Why This Is A Bug

In OpenAI's streaming API, the first chunk typically contains `delta.role = "assistant"` to indicate the responder role, while subsequent chunks have `delta.role = None` since the role doesn't change. The current implementation on line 943:

```python
role = choice.delta.role
```

unconditionally overwrites `role` on every chunk, causing the correct role from the first chunk to be replaced by `None` from subsequent chunks.

This results in incorrect metadata in the combined response, potentially breaking downstream code that depends on the `role` field being correctly set to `"assistant"`.

## Fix

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