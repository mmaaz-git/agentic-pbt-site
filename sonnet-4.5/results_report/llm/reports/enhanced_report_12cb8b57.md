# Bug Report: llm_combine_chunks Role Field Lost When Combining Streaming Chunks

**Target**: `llm.default_plugins.openai_models.combine_chunks`
**Severity**: Medium
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `combine_chunks` function incorrectly overwrites the `role` field with `None` when combining streaming chunks, causing the role information from the first chunk to be lost.

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


if __name__ == "__main__":
    test_combine_chunks_preserves_role()
```

<details>

<summary>
**Failing input**: `contents=['', '']` (or any other list with 2+ elements)
</summary>
```
Traceback (most recent call last):
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 35, in <module>
    test_combine_chunks_preserves_role()
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 27, in test_combine_chunks_preserves_role
    def test_combine_chunks_preserves_role(contents):
                   ^^^
  File "/home/npc/miniconda/lib/python3.13/site-packages/hypothesis/core.py", line 2124, in wrapped_test
    raise the_error_hypothesis_found
  File "/home/npc/pbt/agentic-pbt/worker_/60/hypo.py", line 31, in test_combine_chunks_preserves_role
    assert result["role"] == "assistant"
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError
Falsifying example: test_combine_chunks_preserves_role(
    contents=['', ''],  # or any other generated value
)
```
</details>

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


# Test case with two chunks - first has role, second doesn't
chunks = [
    make_chunk("assistant", "Hello"),
    make_chunk(None, " World"),
]

result = combine_chunks(chunks)
print(f"Result: {result}")
print(f"Role: {result['role']}")
print(f"Content: {result['content']}")

# This assertion should pass, but it fails due to the bug
try:
    assert result["role"] == "assistant", f"Expected role='assistant', got role={result['role']}"
    print("PASS: Role preserved correctly")
except AssertionError as e:
    print(f"FAIL: {e}")
```

<details>

<summary>
Role field incorrectly becomes None instead of preserving "assistant"
</summary>
```
Result: {'content': 'Hello World', 'role': None, 'finish_reason': None, 'usage': {}}
Role: None
Content: Hello World
FAIL: Expected role='assistant', got role=None
```
</details>

## Why This Is A Bug

This bug violates the expected behavior of OpenAI's streaming API implementation. According to OpenAI's API specification, the role field is only transmitted in the first chunk of a streaming response, with subsequent chunks having `delta.role = None` or the field omitted entirely. The function is designed to combine these chunks into a single response, but the current implementation at line 943 unconditionally overwrites the role variable with each chunk's delta.role value, resulting in the final role being None instead of preserving the initial "assistant" role. This causes data loss - the role information that should be preserved from the first chunk is discarded, breaking any downstream code that depends on knowing the message role (e.g., for conversation context, logging, or UI display).

## Relevant Context

The `combine_chunks` function is located at line 919 in `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py`. It is used by the AsyncChat class (line 832) and other locations for processing streaming responses from OpenAI's API.

OpenAI's streaming response format follows this pattern:
- **First chunk**: `{"delta": {"role": "assistant", "content": ""}}`
- **Subsequent chunks**: `{"delta": {"content": "text fragment"}}` (role is None or absent)
- **Final chunk**: `{"delta": {}, "finish_reason": "stop"}`

The function correctly concatenates content and preserves finish_reason, but fails to preserve the role due to the unconditional assignment on line 943.

Package information:
- Package: llm v0.27.1
- Author: Simon Willison
- Repository: https://github.com/simonw/llm

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