#!/usr/bin/env python3
"""
Minimal reproduction of the pandas.io.clipboard timeout bug.
Demonstrates that waitForPaste() and waitForNewPaste() do not respect
timeout values <= 0.01 seconds.
"""

import time
import pandas.io.clipboard as clipboard

# Mock the paste function to always return empty string
# This simulates an empty clipboard that never gets filled
original_paste = clipboard.paste
def mock_empty_paste():
    return ""
clipboard.paste = mock_empty_paste

print("Testing waitForPaste() with various timeout values:")
print("=" * 50)

# Test with timeout=0 (should timeout immediately)
print("\nTest 1: timeout=0 (should timeout immediately)")
start = time.time()
try:
    clipboard.waitForPaste(timeout=0)
    print("ERROR: Should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")
    print(f"Expected: <0.005s, Actual: {elapsed:.4f}s")
    if elapsed > 0.005:
        print("BUG CONFIRMED: Waited longer than expected")

# Test with timeout=-1 (negative, should timeout immediately)
print("\nTest 2: timeout=-1 (negative, should timeout immediately)")
start = time.time()
try:
    clipboard.waitForPaste(timeout=-1)
    print("ERROR: Should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")
    print(f"Expected: <0.005s, Actual: {elapsed:.4f}s")
    if elapsed > 0.005:
        print("BUG CONFIRMED: Waited longer than expected")

# Test with timeout=0.001 (1ms, should timeout after ~1ms)
print("\nTest 3: timeout=0.001 (should timeout after ~0.001s)")
start = time.time()
try:
    clipboard.waitForPaste(timeout=0.001)
    print("ERROR: Should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")
    print(f"Expected: ~0.001s, Actual: {elapsed:.4f}s")
    if elapsed > 0.005:
        print("BUG CONFIRMED: Waited much longer than expected")

print("\n" + "=" * 50)
print("Testing waitForNewPaste() with various timeout values:")
print("=" * 50)

# Mock to return a constant non-empty string for waitForNewPaste
def mock_constant_paste():
    return "constant text"
clipboard.paste = mock_constant_paste

# Test waitForNewPaste with timeout=0
print("\nTest 4: waitForNewPaste with timeout=0")
start = time.time()
try:
    clipboard.waitForNewPaste(timeout=0)
    print("ERROR: Should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")
    print(f"Expected: <0.005s, Actual: {elapsed:.4f}s")
    if elapsed > 0.005:
        print("BUG CONFIRMED: Waited longer than expected")

# Test waitForNewPaste with timeout=-1
print("\nTest 5: waitForNewPaste with timeout=-1")
start = time.time()
try:
    clipboard.waitForNewPaste(timeout=-1)
    print("ERROR: Should have raised PyperclipTimeoutException")
except clipboard.PyperclipTimeoutException as e:
    elapsed = time.time() - start
    print(f"Timed out after {elapsed:.4f}s")
    print(f"Expected: <0.005s, Actual: {elapsed:.4f}s")
    if elapsed > 0.005:
        print("BUG CONFIRMED: Waited longer than expected")

print("\n" + "=" * 50)
print("Summary: The bug exists because the timeout check occurs")
print("AFTER time.sleep(0.01), causing a minimum wait of ~0.01s")
print("regardless of the timeout value specified.")