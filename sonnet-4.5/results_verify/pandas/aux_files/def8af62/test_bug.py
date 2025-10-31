#!/usr/bin/env python3
"""Test script to reproduce the NaN/infinity timeout bug in pandas.io.clipboard"""

import time
import signal
import math
from unittest.mock import patch
import pandas.io.clipboard as clip

def test_nan_timeout():
    """Test that NaN timeout causes hanging"""
    print("Testing NaN timeout...")

    with patch.object(clip, 'paste', return_value=''):
        start = time.time()
        timeout_value = float('nan')

        # Show what happens with NaN comparisons
        print(f"  timeout_value = {timeout_value}")
        print(f"  start = {start}")
        print(f"  start + timeout_value = {start + timeout_value}")
        print(f"  time.time() > start + timeout_value = {time.time() > start + timeout_value}")
        print(f"  Result: Comparison always returns False, will hang forever")

        # Set up a timeout handler to prevent actual hanging
        def timeout_handler(signum, frame):
            raise TimeoutError("Function hung as expected - NaN timeout doesn't work!")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)  # 2 second timeout

        try:
            print(f"  Calling waitForPaste(timeout={timeout_value})...")
            result = clip.waitForPaste(timeout=timeout_value)
            print(f"  ERROR: Function returned unexpectedly: {result}")
        except TimeoutError as e:
            print(f"  CONFIRMED BUG: {e}")
        finally:
            signal.alarm(0)

def test_infinity_timeout():
    """Test that infinity timeout causes hanging"""
    print("\nTesting infinity timeout...")

    with patch.object(clip, 'paste', return_value=''):
        start = time.time()
        timeout_value = float('inf')

        # Show what happens with infinity comparisons
        print(f"  timeout_value = {timeout_value}")
        print(f"  start = {start}")
        print(f"  start + timeout_value = {start + timeout_value}")
        print(f"  time.time() > start + timeout_value = {time.time() > start + timeout_value}")
        print(f"  Result: time.time() is never greater than infinity, will hang forever")

        # Set up a timeout handler to prevent actual hanging
        def timeout_handler(signum, frame):
            raise TimeoutError("Function hung as expected - infinity timeout doesn't work!")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)  # 2 second timeout

        try:
            print(f"  Calling waitForPaste(timeout={timeout_value})...")
            result = clip.waitForPaste(timeout=timeout_value)
            print(f"  ERROR: Function returned unexpectedly: {result}")
        except TimeoutError as e:
            print(f"  CONFIRMED BUG: {e}")
        finally:
            signal.alarm(0)

def test_waitForNewPaste_nan():
    """Test that waitForNewPaste also has the same bug with NaN"""
    print("\nTesting waitForNewPaste with NaN timeout...")

    with patch.object(clip, 'paste', return_value='original'):
        start = time.time()
        timeout_value = float('nan')

        # Set up a timeout handler to prevent actual hanging
        def timeout_handler(signum, frame):
            raise TimeoutError("waitForNewPaste hung with NaN timeout!")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(2)  # 2 second timeout

        try:
            print(f"  Calling waitForNewPaste(timeout={timeout_value})...")
            result = clip.waitForNewPaste(timeout=timeout_value)
            print(f"  ERROR: Function returned unexpectedly: {result}")
        except TimeoutError as e:
            print(f"  CONFIRMED BUG: {e}")
        finally:
            signal.alarm(0)

def test_negative_timeout():
    """Test that negative timeout also causes issues"""
    print("\nTesting negative timeout...")

    with patch.object(clip, 'paste', return_value=''):
        timeout_value = -1.0

        print(f"  Calling waitForPaste(timeout={timeout_value})...")
        try:
            result = clip.waitForPaste(timeout=timeout_value)
            print(f"  Function returned: {result}")
        except clip.PyperclipTimeoutException as e:
            print(f"  Raised PyperclipTimeoutException immediately: {e}")

def test_normal_timeout():
    """Test that normal timeout works correctly"""
    print("\nTesting normal timeout (should work correctly)...")

    with patch.object(clip, 'paste', return_value=''):
        timeout_value = 0.5

        print(f"  Calling waitForPaste(timeout={timeout_value})...")
        start = time.time()
        try:
            result = clip.waitForPaste(timeout=timeout_value)
            print(f"  ERROR: Function returned unexpectedly: {result}")
        except clip.PyperclipTimeoutException as e:
            elapsed = time.time() - start
            print(f"  Correctly raised timeout after {elapsed:.2f} seconds: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing pandas.io.clipboard timeout bug")
    print("=" * 60)

    test_nan_timeout()
    test_infinity_timeout()
    test_waitForNewPaste_nan()
    test_negative_timeout()
    test_normal_timeout()

    print("\n" + "=" * 60)
    print("CONCLUSION: NaN and infinity timeouts cause infinite loops!")
    print("=" * 60)