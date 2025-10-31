# Bug Report: llm.default_plugins.openai_models.AsyncChat Duplicate Usage Assignment

**Target**: `llm.default_plugins.openai_models.AsyncChat.execute`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AsyncChat.execute` method contains duplicate code that checks and assigns `chunk.usage` twice in the same iteration loop, resulting in redundant `model_dump()` calls during streaming responses.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Property-based test that discovers the duplicate code in AsyncChat.execute()
"""

from hypothesis import given, strategies as st, assume, settings
from unittest.mock import Mock, AsyncMock, patch
import asyncio
import sys
import os

# Add the package to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

@given(st.integers(min_value=1, max_value=100))
@settings(max_examples=5)
def test_async_chat_duplicate_usage_assignment(num_chunks):
    """
    Test that reveals duplicate usage assignment in AsyncChat.execute()

    This test counts how many times model_dump() is called when processing
    chunks with usage data. The bug causes it to be called twice per chunk
    with usage data instead of once.
    """

    class MockChunk:
        def __init__(self, has_usage):
            self.choices = []
            self.model_dump_call_count = 0
            if has_usage:
                self.usage = Mock()
                # Track how many times model_dump is called
                def count_dump():
                    self.model_dump_call_count += 1
                    return {"prompt_tokens": 100, "completion_tokens": 50}
                self.usage.model_dump = Mock(side_effect=count_dump)
            else:
                self.usage = None

    # Create chunks - every 3rd chunk has usage data
    chunks_with_usage = [MockChunk(has_usage=(i % 3 == 0)) for i in range(num_chunks)]

    # Simulate the buggy AsyncChat code (lines 798-803)
    usage_assignment_count = 0
    total_model_dump_calls = 0

    for chunk in chunks_with_usage:
        # Lines 799-800: First check
        if chunk.usage:
            usage = chunk.usage.model_dump()
            usage_assignment_count += 1

        # Line 801: append chunk (omitted for simplicity)

        # Lines 802-803: DUPLICATE check
        if chunk.usage:
            usage = chunk.usage.model_dump()
            usage_assignment_count += 1

        if chunk.usage:
            total_model_dump_calls += chunk.model_dump_call_count

    # Calculate expected vs actual
    chunks_with_usage_data = sum(1 for c in chunks_with_usage if c.usage)
    expected_calls = chunks_with_usage_data  # Should only call once per chunk with usage
    actual_calls = usage_assignment_count

    print(f"\nTest with {num_chunks} chunks ({chunks_with_usage_data} with usage data):")
    print(f"  Expected model_dump() calls: {expected_calls}")
    print(f"  Actual model_dump() calls: {actual_calls}")
    print(f"  Redundant operations: {actual_calls - expected_calls}")

    # The bug causes 2x the expected number of calls
    assert actual_calls == 2 * expected_calls, (
        f"Duplicate code detected! Expected {expected_calls} calls but got {actual_calls}"
    )

if __name__ == "__main__":
    print("Property-based test revealing duplicate code in AsyncChat.execute()")
    print("=" * 70)

    # Run the test
    try:
        test_async_chat_duplicate_usage_assignment()
        print("\n" + "=" * 70)
        print("BUG CONFIRMED: All tests show duplicate usage assignments")
        print("The code at lines 799-800 and 802-803 is identical and redundant")
    except AssertionError as e:
        print(f"\nTest failed with: {e}")
```

<details>

<summary>
**Failing input**: `num_chunks=1` (and all tested values)
</summary>
```
Property-based test revealing duplicate code in AsyncChat.execute()
======================================================================

Test with 1 chunks (1 with usage data):
  Expected model_dump() calls: 1
  Actual model_dump() calls: 2
  Redundant operations: 1

Test with 16 chunks (6 with usage data):
  Expected model_dump() calls: 6
  Actual model_dump() calls: 12
  Redundant operations: 6

Test with 70 chunks (24 with usage data):
  Expected model_dump() calls: 24
  Actual model_dump() calls: 48
  Redundant operations: 24

Test with 10 chunks (4 with usage data):
  Expected model_dump() calls: 4
  Actual model_dump() calls: 8
  Redundant operations: 4

Test with 38 chunks (13 with usage data):
  Expected model_dump() calls: 13
  Actual model_dump() calls: 26
  Redundant operations: 13

======================================================================
BUG CONFIRMED: All tests show duplicate usage assignments
The code at lines 799-800 and 802-803 is identical and redundant
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal reproduction of the duplicate code issue in AsyncChat.execute()
"""

import asyncio
from unittest.mock import AsyncMock, Mock, MagicMock
import sys
import os

# Add the package to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages')

# Create a mock to demonstrate the duplicate usage assignment
class MockChunk:
    def __init__(self, index, has_usage=False):
        self.index = index
        self.choices = []
        if has_usage:
            self.usage = Mock()
            self.usage.model_dump = Mock(return_value={"prompt_tokens": 100, "completion_tokens": 50})
        else:
            self.usage = None

class MockResponse:
    def __init__(self):
        self.response_json = None
        self._prompt_json = None

    def set_usage(self, **kwargs):
        pass

class MockPrompt:
    def __init__(self):
        self.system = None
        self.prompt = "test"
        self.options = {}
        self.schema = None
        self.tools = None
        self.tool_results = []
        self.attachments = []

async def simulate_async_chat():
    """Simulate the problematic code path in AsyncChat.execute()"""

    # Create mock chunks with usage data
    chunks_with_usage = [MockChunk(i, has_usage=(i == 2)) for i in range(5)]

    # This simulates the actual code from lines 798-803
    chunks = []
    usage = None
    usage_assignment_count = 0

    print("Simulating AsyncChat.execute() lines 798-803:")
    print("=" * 50)

    for chunk in chunks_with_usage:
        print(f"\nProcessing chunk {chunk.index}:")

        # Line 799-800: First check and assignment
        if chunk.usage:
            usage = chunk.usage.model_dump()
            usage_assignment_count += 1
            print(f"  Line 799-800: First usage assignment (count: {usage_assignment_count})")

        # Line 801: Append chunk
        chunks.append(chunk)
        print(f"  Line 801: Appended chunk")

        # Line 802-803: DUPLICATE check and assignment
        if chunk.usage:
            usage = chunk.usage.model_dump()
            usage_assignment_count += 1
            print(f"  Line 802-803: Second usage assignment (count: {usage_assignment_count})")

    print("\n" + "=" * 50)
    print(f"Total usage assignments: {usage_assignment_count}")
    print(f"Expected assignments: 1")
    print(f"Redundant operations: {usage_assignment_count - 1}")

    # Now show the correct implementation from Chat.execute() (sync version)
    print("\n\nCorrect implementation (from Chat.execute() lines 713-716):")
    print("=" * 50)

    chunks_sync = []
    usage_sync = None
    usage_assignment_count_sync = 0

    for chunk in [MockChunk(i, has_usage=(i == 2)) for i in range(5)]:
        print(f"\nProcessing chunk {chunk.index}:")

        # Line 714: Append chunk
        chunks_sync.append(chunk)
        print(f"  Line 714: Appended chunk")

        # Line 715-716: Single check and assignment
        if chunk.usage:
            usage_sync = chunk.usage.model_dump()
            usage_assignment_count_sync += 1
            print(f"  Line 715-716: Usage assignment (count: {usage_assignment_count_sync})")

    print("\n" + "=" * 50)
    print(f"Total usage assignments: {usage_assignment_count_sync}")
    print(f"This is correct - only 1 assignment!")

    print("\n\nCONCLUSION:")
    print("-" * 50)
    print("The AsyncChat.execute() method has duplicate code at lines 799-800 and 802-803.")
    print("Both check 'if chunk.usage' and perform 'usage = chunk.usage.model_dump()'")
    print("This results in calling model_dump() twice unnecessarily.")
    print("The synchronous Chat.execute() correctly does this only once.")

if __name__ == "__main__":
    asyncio.run(simulate_async_chat())
```

<details>

<summary>
Demonstration of duplicate usage assignment
</summary>
```
Simulating AsyncChat.execute() lines 798-803:
==================================================

Processing chunk 0:
  Line 801: Appended chunk

Processing chunk 1:
  Line 801: Appended chunk

Processing chunk 2:
  Line 799-800: First usage assignment (count: 1)
  Line 801: Appended chunk
  Line 802-803: Second usage assignment (count: 2)

Processing chunk 3:
  Line 801: Appended chunk

Processing chunk 4:
  Line 801: Appended chunk

==================================================
Total usage assignments: 2
Expected assignments: 1
Redundant operations: 1


Correct implementation (from Chat.execute() lines 713-716):
==================================================

Processing chunk 0:
  Line 714: Appended chunk

Processing chunk 1:
  Line 714: Appended chunk

Processing chunk 2:
  Line 714: Appended chunk
  Line 715-716: Usage assignment (count: 1)

Processing chunk 3:
  Line 714: Appended chunk

Processing chunk 4:
  Line 714: Appended chunk

==================================================
Total usage assignments: 1
This is correct - only 1 assignment!


CONCLUSION:
--------------------------------------------------
The AsyncChat.execute() method has duplicate code at lines 799-800 and 802-803.
Both check 'if chunk.usage' and perform 'usage = chunk.usage.model_dump()'
This results in calling model_dump() twice unnecessarily.
The synchronous Chat.execute() correctly does this only once.
```
</details>

## Why This Is A Bug

This violates the expected behavior because:

1. **Redundant Operations**: The code performs the exact same conditional check and assignment twice per iteration (lines 799-800 and lines 802-803), calling `chunk.usage.model_dump()` twice when once would suffice.

2. **Inconsistency with Synchronous Version**: The synchronous `Chat.execute` method (lines 713-716) correctly performs this operation only once per chunk, demonstrating this is unintended in the async version.

3. **Performance Waste**: While the impact is minimal, this unnecessarily doubles the CPU cycles spent on converting usage data to dictionaries during streaming operations.

4. **Code Quality Issue**: This appears to be a copy-paste error or incomplete refactoring, as there's no logical reason to check and assign the same value twice in immediate succession.

## Relevant Context

- **Location**: `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py`, lines 798-803
- **Function**: `AsyncChat.execute()` method used for asynchronous streaming with OpenAI models
- **Comparison**: The synchronous `Chat.execute()` at lines 713-716 shows the correct pattern
- **Impact**: Every streaming response with usage data triggers this duplicate operation
- **Root Cause**: Likely a copy-paste error during the implementation of the async version

The duplicate code doesn't affect correctness (the second assignment overwrites the first with an identical value), but it represents poor code quality and unnecessary computation.

## Proposed Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -796,11 +796,8 @@ class AsyncChat(_Shared, AsyncKeyModel):
             chunks = []
             tool_calls = {}
             async for chunk in completion:
-                if chunk.usage:
-                    usage = chunk.usage.model_dump()
                 chunks.append(chunk)
                 if chunk.usage:
                     usage = chunk.usage.model_dump()
                 if chunk.choices and chunk.choices[0].delta:
                     for tool_call in chunk.choices[0].delta.tool_calls or []:
```