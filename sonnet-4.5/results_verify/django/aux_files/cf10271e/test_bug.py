#!/usr/bin/env python3
"""Test to reproduce the Django email backend exception masking bug."""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.core.mail.backends.base import BaseEmailBackend

# Test 1: Property-based test from bug report
def test_base_context_manager_exception_handling():
    class FailingBackend(BaseEmailBackend):
        def open(self):
            raise ValueError("Open failed")

        def close(self):
            raise RuntimeError("Close failed")

        def send_messages(self, email_messages):
            return 0

    backend = FailingBackend()
    try:
        with backend:
            pass
        assert False, "Should have raised an exception"
    except ValueError:
        print("TEST 1 PASSED: Caught ValueError as expected")
    except RuntimeError:
        print("TEST 1 FAILED: RuntimeError from close() masked the ValueError from open()")
    except Exception as e:
        print(f"TEST 1 ERROR: Unexpected exception: {type(e).__name__}: {e}")

# Test 2: Reproduce the bug directly
def test_reproduce_bug():
    class FailingBackend(BaseEmailBackend):
        def open(self):
            raise ValueError("Open failed")

        def close(self):
            raise RuntimeError("Close failed")

        def send_messages(self, email_messages):
            return 0

    backend = FailingBackend()
    print("\nTEST 2: Direct reproduction")
    try:
        with backend:
            pass
    except Exception as e:
        print(f"Caught: {type(e).__name__}: {e}")
        print(f"Expected: ValueError: Open failed")
        print(f"Actual matches bug report: {type(e).__name__ == 'RuntimeError'}")

# Test 3: Test with close() that doesn't raise
def test_with_safe_close():
    class SafeCloseBackend(BaseEmailBackend):
        def open(self):
            raise ValueError("Open failed")

        def close(self):
            print("Close called without exception")

        def send_messages(self, email_messages):
            return 0

    backend = SafeCloseBackend()
    print("\nTEST 3: With safe close()")
    try:
        with backend:
            pass
    except Exception as e:
        print(f"Caught: {type(e).__name__}: {e}")
        print(f"Correct - original exception preserved when close() doesn't raise")

# Test 4: Test normal operation (no exceptions)
def test_normal_operation():
    class WorkingBackend(BaseEmailBackend):
        def open(self):
            print("Open successful")

        def close(self):
            print("Close successful")

        def send_messages(self, email_messages):
            return 0

    backend = WorkingBackend()
    print("\nTEST 4: Normal operation")
    try:
        with backend:
            print("Inside context manager")
        print("No exceptions - correct behavior")
    except Exception as e:
        print(f"Unexpected exception: {type(e).__name__}: {e}")

if __name__ == "__main__":
    test_base_context_manager_exception_handling()
    test_reproduce_bug()
    test_with_safe_close()
    test_normal_operation()