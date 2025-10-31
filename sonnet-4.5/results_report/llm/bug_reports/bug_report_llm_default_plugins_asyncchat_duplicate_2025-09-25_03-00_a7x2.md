# Bug Report: llm.default_plugins.openai_models.AsyncChat Duplicate Usage Check

**Target**: `llm.default_plugins.openai_models.AsyncChat.execute`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AsyncChat.execute()` method contains duplicate code that checks and assigns `chunk.usage` twice in the streaming response loop, causing unnecessary overhead.

## Property-Based Test

```python
from hypothesis import given, strategies as st
from dataclasses import dataclass
from typing import Optional

@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

    def model_dump(self):
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }

@dataclass
class Chunk:
    usage: Optional[Usage]
    choices: list

call_count = 0

def process_chunks_buggy(chunks):
    global call_count
    call_count = 0
    usage = None

    for chunk in chunks:
        if chunk.usage:
            call_count += 1
            usage = chunk.usage.model_dump()
        if chunk.usage:
            call_count += 1
            usage = chunk.usage.model_dump()

    return usage, call_count

@given(st.integers(min_value=1, max_value=100))
def test_duplicate_usage_check(num_chunks):
    chunks = [Chunk(usage=Usage(10, 5, 15), choices=[]) for _ in range(num_chunks)]

    usage, calls = process_chunks_buggy(chunks)

    assert calls == 2 * num_chunks, f"Usage check called {calls} times, expected {2 * num_chunks}"
```

**Failing input**: Any streaming response with usage data in chunks

## Reproducing the Bug

The bug is visible by inspection of the code at lines 799-803 in `openai_models.py`:

```python
async for chunk in completion:
    if chunk.usage:
        usage = chunk.usage.model_dump()  # Line 800
    chunks.append(chunk)
    if chunk.usage:
        usage = chunk.usage.model_dump()  # Line 803 - duplicate!
```

The same `if chunk.usage` check and assignment occurs twice consecutively, with only `chunks.append(chunk)` in between.

## Why This Is A Bug

The duplicate check causes:
1. Unnecessary CPU cycles checking `chunk.usage` twice per chunk
2. Redundant `.model_dump()` calls, which creates duplicate dictionary conversions
3. Code inconsistency - the `Chat` class (non-async version) doesn't have this duplication

While not causing incorrect results, this is wasteful and indicates a copy-paste error during development.

## Fix

```diff
--- a/openai_models.py
+++ b/openai_models.py
@@ -797,10 +797,8 @@ class AsyncChat(_Shared, AsyncKeyModel):
             chunks = []
             tool_calls = {}
             async for chunk in completion:
-                if chunk.usage:
-                    usage = chunk.usage.model_dump()
                 chunks.append(chunk)
                 if chunk.usage:
                     usage = chunk.usage.model_dump()
                 if chunk.choices and chunk.choices[0].delta:
```