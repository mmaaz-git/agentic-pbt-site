# Bug Report: combine_chunks Role Loss

**Target**: `llm.default_plugins.openai_models.combine_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `combine_chunks` function incorrectly handles the `role` field when combining streaming chunks, causing the role to be lost when subsequent chunks have `role=None`.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from llm.default_plugins.openai_models import combine_chunks
from unittest.mock import Mock


def make_chunk(role, content):
    chunk = Mock()
    chunk.usage = None
    chunk.id = None
    chunk.object = None
    chunk.model = None
    chunk.created = None
    chunk.index = None

    choice = Mock()
    delta = Mock()
    delta.role = role
    delta.content = content
    choice.delta = delta
    choice.finish_reason = None
    choice.logprobs = None
    chunk.choices = [choice]
    return chunk


@given(st.lists(st.text(), min_size=2, max_size=10))
def test_combine_chunks_preserves_role(contents):
    chunks = [make_chunk("assistant" if i == 0 else None, content)
              for i, content in enumerate(contents)]
    result = combine_chunks(chunks)
    assert result["role"] == "assistant"
```

**Failing input**: `contents=["Hello", "World"]` (or any list with 2+ elements)

## Reproducing the Bug

```python
from llm.default_plugins.openai_models import combine_chunks
from unittest.mock import Mock


def make_chunk(role, content):
    chunk = Mock()
    chunk.usage = None
    chunk.id = chunk.object = chunk.model = chunk.created = chunk.index = None
    choice = Mock()
    delta = Mock()
    delta.role = role
    delta.content = content
    choice.delta = delta
    choice.finish_reason = None
    choice.logprobs = None
    chunk.choices = [choice]
    return chunk


chunks = [
    make_chunk("assistant", "Hello"),
    make_chunk(None, " World"),
]

result = combine_chunks(chunks)
print(f"Role: {result['role']}")
assert result["role"] == "assistant"
```

## Why This Is A Bug

In OpenAI's streaming API, the `role` field is only present in the first chunk's delta, while subsequent chunks have `delta.role = None`. The current implementation overwrites the `role` variable in each iteration (line 943), causing the final value to be `None` instead of preserving the role from the first chunk.

## Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -940,7 +940,8 @@ def combine_chunks(chunks: List) -> dict:
                 continue
-            role = choice.delta.role
+            if choice.delta.role is not None:
+                role = choice.delta.role
             if choice.delta.content is not None:
                 content += choice.delta.content
             if choice.finish_reason is not None:
```