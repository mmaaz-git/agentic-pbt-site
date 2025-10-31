#!/usr/bin/env python3
"""Demonstration of pandas.io.clipboard.waitForPaste bug with non-string values"""

import pandas.io.clipboard as clipboard

# Save original paste function
original_paste = clipboard.paste

# Test case 1: None value
print("Test Case 1: None value")
clipboard.paste = lambda: None
try:
    result = clipboard.waitForPaste(timeout=0.05)
    print(f"Result: {repr(result)}, Type: {type(result).__name__}")
except clipboard.PyperclipTimeoutException as e:
    print(f"Timed out as expected: {e}")

# Test case 2: Integer 0
print("\nTest Case 2: Integer 0")
clipboard.paste = lambda: 0
try:
    result = clipboard.waitForPaste(timeout=0.05)
    print(f"Result: {repr(result)}, Type: {type(result).__name__}")
except clipboard.PyperclipTimeoutException as e:
    print(f"Timed out as expected: {e}")

# Test case 3: Boolean False
print("\nTest Case 3: Boolean False")
clipboard.paste = lambda: False
try:
    result = clipboard.waitForPaste(timeout=0.05)
    print(f"Result: {repr(result)}, Type: {type(result).__name__}")
except clipboard.PyperclipTimeoutException as e:
    print(f"Timed out as expected: {e}")

# Test case 4: Empty list
print("\nTest Case 4: Empty list []")
clipboard.paste = lambda: []
try:
    result = clipboard.waitForPaste(timeout=0.05)
    print(f"Result: {repr(result)}, Type: {type(result).__name__}")
except clipboard.PyperclipTimeoutException as e:
    print(f"Timed out as expected: {e}")

# Test case 5: Empty dict
print("\nTest Case 5: Empty dict {}")
clipboard.paste = lambda: {}
try:
    result = clipboard.waitForPaste(timeout=0.05)
    print(f"Result: {repr(result)}, Type: {type(result).__name__}")
except clipboard.PyperclipTimeoutException as e:
    print(f"Timed out as expected: {e}")

# Restore original paste function
clipboard.paste = original_paste

print("\nExpected behavior: All tests should timeout since no non-empty string exists.")
print("Actual behavior: Functions return non-string values immediately.")