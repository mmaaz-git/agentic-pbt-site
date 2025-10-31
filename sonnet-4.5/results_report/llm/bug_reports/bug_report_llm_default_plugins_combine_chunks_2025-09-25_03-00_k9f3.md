# Bug Report: llm.default_plugins.openai_models.combine_chunks Loses Role

**Target**: `llm.default_plugins.openai_models.combine_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `combine_chunks()` function incorrectly overwrites the role field with None when processing streaming responses, causing response metadata to be lost.

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

@dataclass
class Chunk:
    choices: List[Choice]
    usage: Optional[object] = None

def combine_chunks(chunks: List) -> dict:
    content = ""
    role = None
    finish_reason = None

    for item in chunks:
        for choice in item.choices:
            if not hasattr(choice, "delta"):
                continue
            role = choice.delta.role
            if choice.delta.content is not None:
                content += choice.delta.content
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason

    return {"content": content, "role": role, "finish_reason": finish_reason}

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
```

**Failing input**: Any streaming response with 2+ chunks where only the first chunk contains a role (standard OpenAI streaming behavior)

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

@dataclass
class Chunk:
    choices: List[Choice]

def combine_chunks(chunks: List) -> dict:
    content = ""
    role = None
    finish_reason = None

    for item in chunks:
        for choice in item.choices:
            role = choice.delta.role
            if choice.delta.content is not None:
                content += choice.delta.content

    return {"content": content, "role": role}

chunks = [
    Chunk(choices=[Choice(delta=Delta(role="assistant", content="Hello"), finish_reason=None)]),
    Chunk(choices=[Choice(delta=Delta(role=None, content=" world"), finish_reason=None)]),
]

result = combine_chunks(chunks)
print(result['role'])
```

**Output**: `None` (expected: `"assistant"`)

## Why This Is A Bug

In OpenAI's streaming API, only the first chunk contains the role (e.g., "assistant"), while subsequent chunks have `delta.role = None`. The current code unconditionally assigns `role = choice.delta.role` on every iteration, so when the second chunk is processed, it overwrites the valid role with None.

This causes the combined response to lose the role metadata, which is required for proper conversation tracking and debugging. The role should only be updated when a non-None value is encountered.

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