# Bug Report: llm.default_plugins AsyncChat Duplicate Usage Assignment

**Target**: `llm.default_plugins.openai_models.AsyncChat.execute`
**Severity**: Low
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The `AsyncChat.execute` method contains redundant duplicate code that checks and assigns `chunk.usage` twice consecutively within the same loop iteration, violating the DRY principle and creating an inconsistency with the synchronous `Chat.execute` implementation.

## Property-Based Test

```python
#!/usr/bin/env python3
"""
Hypothesis test to verify consistency between sync and async implementations
"""

import ast
from hypothesis import given, strategies as st, settings
from pathlib import Path

def count_usage_checks_in_loop(source_code: str, class_name: str, method_name: str, is_async: bool) -> int:
    """Count the number of 'if chunk.usage:' checks in a method's main loop"""
    tree = ast.parse(source_code)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for method in node.body:
                if is_async and isinstance(method, ast.AsyncFunctionDef) and method.name == method_name:
                    for stmt in ast.walk(method):
                        if isinstance(stmt, ast.AsyncFor):
                            usage_checks = 0
                            for body_stmt in stmt.body:
                                if isinstance(body_stmt, ast.If):
                                    if (isinstance(body_stmt.test, ast.Attribute) and
                                        isinstance(body_stmt.test.value, ast.Name) and
                                        body_stmt.test.value.id == "chunk" and
                                        body_stmt.test.attr == "usage"):
                                        usage_checks += 1
                            return usage_checks
                elif not is_async and isinstance(method, ast.FunctionDef) and method.name == method_name:
                    for stmt in ast.walk(method):
                        if isinstance(stmt, ast.For):
                            usage_checks = 0
                            for body_stmt in stmt.body:
                                if isinstance(body_stmt, ast.If):
                                    if (isinstance(body_stmt.test, ast.Attribute) and
                                        isinstance(body_stmt.test.value, ast.Name) and
                                        body_stmt.test.value.id == "chunk" and
                                        body_stmt.test.attr == "usage"):
                                        usage_checks += 1
                            return usage_checks
    return -1

@given(st.just(None))  # We don't need random input for this static code analysis
@settings(max_examples=1, deadline=None)
def test_sync_async_consistency(dummy_input):
    """Property: Async and sync implementations should have identical logic patterns"""

    # Read the actual source file
    source_file = Path("/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py")
    with open(source_file, 'r') as f:
        source_code = f.read()

    # Count usage checks in both versions
    sync_checks = count_usage_checks_in_loop(source_code, "Chat", "execute", is_async=False)
    async_checks = count_usage_checks_in_loop(source_code, "AsyncChat", "execute", is_async=True)

    print(f"Sync Chat.execute: {sync_checks} usage check(s)")
    print(f"Async AsyncChat.execute: {async_checks} usage check(s)")

    # Property assertion: Both should have the same number of usage checks
    assert sync_checks == async_checks, (
        f"Inconsistency detected: Sync version has {sync_checks} usage check(s), "
        f"but async version has {async_checks} usage check(s). "
        f"This violates the principle that async/sync versions should have identical logic."
    )

if __name__ == "__main__":
    print("Running Hypothesis test for sync/async consistency...")
    print("=" * 60)
    try:
        test_sync_async_consistency()
        print("\n✓ TEST PASSED: Sync and async implementations are consistent")
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        print("\nThis is a code quality bug:")
        print("- The duplicate usage check in AsyncChat.execute is redundant")
        print("- It differs from the sync Chat.execute implementation")
        print("- While functionally harmless, it violates DRY principle")
        print("- Likely a copy-paste error during development")
```

<details>

<summary>
**Failing input**: `None` (static code analysis)
</summary>
```
Running Hypothesis test for sync/async consistency...
============================================================
Sync Chat.execute: 1 usage check(s)
Async AsyncChat.execute: 2 usage check(s)
Sync Chat.execute: 1 usage check(s)
Async AsyncChat.execute: 2 usage check(s)

✗ TEST FAILED: Inconsistency detected: Sync version has 1 usage check(s), but async version has 2 usage check(s). This violates the principle that async/sync versions should have identical logic.

This is a code quality bug:
- The duplicate usage check in AsyncChat.execute is redundant
- It differs from the sync Chat.execute implementation
- While functionally harmless, it violates DRY principle
- Likely a copy-paste error during development
```
</details>

## Reproducing the Bug

```python
#!/usr/bin/env python3
"""
Minimal test case demonstrating duplicate usage assignment in AsyncChat.execute
"""

import ast
import inspect
from pathlib import Path

# Read the source file
source_file = Path("/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py")
with open(source_file, 'r') as f:
    source_code = f.read()

# Parse the source code
tree = ast.parse(source_code)

# Find AsyncChat class and its execute method
async_duplicate_found = False
async_lines = []

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "AsyncChat":
        for method in node.body:
            if isinstance(method, ast.AsyncFunctionDef) and method.name == "execute":
                # Look for the async for loop
                for stmt in ast.walk(method):
                    if isinstance(stmt, ast.AsyncFor):
                        # Count usage checks in the loop body
                        usage_checks = []
                        for body_stmt in stmt.body:
                            if isinstance(body_stmt, ast.If):
                                # Check if this is testing chunk.usage
                                if (isinstance(body_stmt.test, ast.Attribute) and
                                    isinstance(body_stmt.test.value, ast.Name) and
                                    body_stmt.test.value.id == "chunk" and
                                    body_stmt.test.attr == "usage"):
                                    usage_checks.append(body_stmt.lineno)

                        if len(usage_checks) == 2:
                            async_duplicate_found = True
                            async_lines = usage_checks
                            print("DUPLICATE CODE FOUND in AsyncChat.execute:")
                            print(f"  - First usage check at line {usage_checks[0]}")
                            print(f"  - Second usage check at line {usage_checks[1]}")

# Find Chat class (sync version) for comparison
sync_checks_found = 0
sync_lines = []

for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef) and node.name == "Chat":
        for method in node.body:
            if isinstance(method, ast.FunctionDef) and method.name == "execute":
                # Look for the for loop
                for stmt in ast.walk(method):
                    if isinstance(stmt, ast.For):
                        # Count usage checks in the loop body
                        for body_stmt in stmt.body:
                            if isinstance(body_stmt, ast.If):
                                # Check if this is testing chunk.usage
                                if (isinstance(body_stmt.test, ast.Attribute) and
                                    isinstance(body_stmt.test.value, ast.Name) and
                                    body_stmt.test.value.id == "chunk" and
                                    body_stmt.test.attr == "usage"):
                                    sync_checks_found += 1
                                    sync_lines.append(body_stmt.lineno)

print("\nCOMPARISON with sync Chat.execute:")
print(f"  - Sync version has {sync_checks_found} usage check(s) at line(s): {sync_lines}")
print(f"  - Async version has {len(async_lines)} usage check(s) at line(s): {async_lines}")

# Show the actual duplicate lines
print("\nACTUAL CODE (AsyncChat.execute lines 798-803):")
lines = source_code.split('\n')
for i in range(797, 803):  # 0-indexed, so 797 = line 798
    print(f"  {i+1:4d}: {lines[i]}")

print("\nACTUAL CODE (Chat.execute lines 713-716):")
for i in range(712, 716):  # 0-indexed
    print(f"  {i+1:4d}: {lines[i]}")

print("\nBUG ANALYSIS:")
print("=" * 50)
if async_duplicate_found:
    print("✗ BUG CONFIRMED: AsyncChat.execute contains duplicate usage assignment")
    print("  The same 'if chunk.usage: usage = chunk.usage.model_dump()' appears twice")
    print("  This is redundant and differs from the sync Chat.execute implementation")
    print("\nIMPACT:")
    print("  - Functional: None (code still works correctly)")
    print("  - Performance: Minor (unnecessary duplicate check and assignment)")
    print("  - Code Quality: Violates DRY principle, inconsistent with sync version")
else:
    print("✓ No duplicate found (unexpected)")
```

<details>

<summary>
Duplicate code confirmed at lines 799 and 802
</summary>
```
DUPLICATE CODE FOUND in AsyncChat.execute:
  - First usage check at line 799
  - Second usage check at line 802

COMPARISON with sync Chat.execute:
  - Sync version has 1 usage check(s) at line(s): [715]
  - Async version has 2 usage check(s) at line(s): [799, 802]

ACTUAL CODE (AsyncChat.execute lines 798-803):
   798:             async for chunk in completion:
   799:                 if chunk.usage:
   800:                     usage = chunk.usage.model_dump()
   801:                 chunks.append(chunk)
   802:                 if chunk.usage:
   803:                     usage = chunk.usage.model_dump()

ACTUAL CODE (Chat.execute lines 713-716):
   713:             for chunk in completion:
   714:                 chunks.append(chunk)
   715:                 if chunk.usage:
   716:                     usage = chunk.usage.model_dump()

BUG ANALYSIS:
==================================================
✗ BUG CONFIRMED: AsyncChat.execute contains duplicate usage assignment
  The same 'if chunk.usage: usage = chunk.usage.model_dump()' appears twice
  This is redundant and differs from the sync Chat.execute implementation

IMPACT:
  - Functional: None (code still works correctly)
  - Performance: Minor (unnecessary duplicate check and assignment)
  - Code Quality: Violates DRY principle, inconsistent with sync version
```
</details>

## Why This Is A Bug

This code violates fundamental software engineering principles and represents an unintentional error:

1. **Duplicate Logic**: Lines 799-800 and 802-803 in `AsyncChat.execute` perform the exact same operation - checking if `chunk.usage` exists and assigning `usage = chunk.usage.model_dump()`. The second check immediately follows the first with no intervening code that could modify `chunk.usage`, making it completely redundant.

2. **Inconsistency with Sync Implementation**: The synchronous `Chat.execute` method (lines 713-716) correctly performs this check only once. In standard library design, async and sync versions of the same functionality should have identical logic patterns, differing only in their use of async/await syntax. This unexplained divergence violates that principle.

3. **No Functional Justification**: There is no documentation or architectural reason for processing usage data twice. The OpenAI API streaming responses don't require duplicate processing, and the variable `usage` simply gets assigned the same value twice.

4. **Clear Copy-Paste Error Pattern**: The code structure suggests someone copied the usage check lines but accidentally pasted them in two locations within the loop body, a common development mistake.

5. **Performance Waste**: While minor, this unnecessarily doubles the CPU cycles spent on:
   - Evaluating the `chunk.usage` condition
   - Calling `chunk.usage.model_dump()` when usage data exists
   - Variable assignment operations

## Relevant Context

- **Source Location**: `/home/npc/pbt/agentic-pbt/envs/llm_env/lib/python3.13/site-packages/llm/default_plugins/openai_models.py`
- **Project**: The `llm` library by Simon Willison (https://github.com/simonw/llm)
- **Classes Involved**: `AsyncChat` (async implementation) and `Chat` (sync implementation)
- **Method**: `execute()` - handles streaming completions from OpenAI-compatible APIs
- **Impact Scope**: Only affects async streaming completions when usage data is present

The duplicate code has no functional impact since:
- The second assignment overwrites the first with identical data
- No side effects occur from `model_dump()`
- The final `usage` value remains correct

However, this represents poor code quality that should be fixed to maintain consistency and clarity.

## Proposed Fix

```diff
--- a/llm/default_plugins/openai_models.py
+++ b/llm/default_plugins/openai_models.py
@@ -796,11 +796,9 @@ class AsyncChat(_Shared, AsyncKeyModel):
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
                         if tool_call.function.arguments is None:
```