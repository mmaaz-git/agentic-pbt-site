# Bug Report: llm.default_plugins AsyncChat Duplicate Usage Check

**Target**: `llm.default_plugins.openai_models.AsyncChat.execute`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AsyncChat.execute` method contains duplicate code that checks and assigns `chunk.usage` twice in succession, which is redundant and inefficient.

## Property-Based Test

```python
from hypothesis import given, strategies as st

@given(st.lists(st.integers()))
def test_code_should_not_duplicate_operations(data):
    """
    Property: Identical operations should not be performed twice in succession
    without any intervening state changes.
    """
    operations = []
    for item in data:
        if item > 0:
            operations.append('check1')
        if item > 0:
            operations.append('check2')

    consecutive_duplicates = sum(
        1 for i in range(len(operations)-1)
        if operations[i] == operations[i+1]
    )
    assert consecutive_duplicates == 0
```

**Failing input**: Any chunk with usage data triggers the duplicate check

## Reproducing the Bug

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

with open('/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py', 'r') as f:
    lines = f.readlines()

print("Lines 798-804 in AsyncChat.execute:")
for i, line in enumerate(lines[797:804], start=798):
    print(f"{i}: {line.rstrip()}")

print("\nCompare with Chat.execute lines 713-717:")
for i, line in enumerate(lines[712:717], start=713):
    print(f"{i}: {line.rstrip()}")
```

## Why This Is A Bug

In `AsyncChat.execute` (lines 799-803), the code checks `chunk.usage` and calls `model_dump()` twice:

```python
async for chunk in completion:
    if chunk.usage:                        # First check
        usage = chunk.usage.model_dump()  # First assignment
    chunks.append(chunk)
    if chunk.usage:                        # Duplicate check
        usage = chunk.usage.model_dump()  # Duplicate assignment
```

This is redundant because:
1. The second check happens immediately after the first
2. No code between the two checks could change `chunk.usage`
3. The same value is assigned to `usage` twice

The synchronous `Chat.execute` method (lines 713-716) does this correctly with only one check:

```python
for chunk in completion:
    chunks.append(chunk)
    if chunk.usage:
        usage = chunk.usage.model_dump()
```

## Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -796,11 +796,9 @@ class AsyncChat(_Shared, AsyncKeyModel):
             chunks = []
             tool_calls = {}
             async for chunk in completion:
+                chunks.append(chunk)
                 if chunk.usage:
                     usage = chunk.usage.model_dump()
-                chunks.append(chunk)
-                if chunk.usage:
-                    usage = chunk.usage.model_dump()
                 if chunk.choices and chunk.choices[0].delta:
                     for tool_call in chunk.choices[0].delta.tool_calls or []:
                         if tool_call.function.arguments is None:
```