#!/usr/bin/env python3
"""Check how Python's standard library handles similar patterns."""

import contextlib
import io

# Test how contextlib.closing handles exceptions
print("Testing contextlib.closing behavior:")

class FailOnClose:
    def close(self):
        raise RuntimeError("Close failed")

class FailOnEnter:
    def __enter__(self):
        raise ValueError("Enter failed")
    def __exit__(self, *args):
        raise RuntimeError("Exit failed")

# Test 1: contextlib.closing doesn't help here since it's for __exit__ only
try:
    with contextlib.closing(FailOnClose()) as f:
        print("Inside context")
except RuntimeError as e:
    print(f"Caught from closing: {e}")

# Test 2: Check how file objects handle this
print("\nTesting file object behavior:")
# Files handle this differently - they don't call close() in __enter__ if open fails

# Test 3: Check database connections patterns
print("\nChecking common patterns in context managers...")

# Most context managers follow this pattern:
# 1. Resource acquisition in __init__ or a factory method
# 2. __enter__ just returns self
# 3. __exit__ does cleanup

# The Django pattern is unusual because:
# - open() is called in __enter__ (not __init__)
# - close() is called for cleanup if open() fails
# - But close() exception can mask open() exception