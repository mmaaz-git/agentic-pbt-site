#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.util import capitalize_first_letter

# Test with German ß character
s = 'ß'
result = capitalize_first_letter(s)

print(f"Input: {repr(s)} (length {len(s)})")
print(f"Output: {repr(result)} (length {len(result)})")
print(f"s[1:] = {repr(s[1:])}")
print(f"result[1:] = {repr(result[1:])}")
print()

# Verify the assertions from the bug report
try:
    assert len(result) == 2, f"Expected length 2, got {len(result)}"
    print("✓ len(result) == 2")
except AssertionError as e:
    print(f"✗ {e}")

try:
    assert result == 'SS', f"Expected 'SS', got {repr(result)}"
    print("✓ result == 'SS'")
except AssertionError as e:
    print(f"✗ {e}")

try:
    assert s[1:] == '', f"Expected s[1:] to be '', got {repr(s[1:])}"
    print("✓ s[1:] == ''")
except AssertionError as e:
    print(f"✗ {e}")

try:
    assert result[1:] == 'S', f"Expected result[1:] to be 'S', got {repr(result[1:])}"
    print("✓ result[1:] == 'S'")
except AssertionError as e:
    print(f"✗ {e}")

print("\nTesting that result[1:] == s[1:] (the bug claim):")
try:
    assert result[1:] == s[1:], f"result[1:] ({repr(result[1:])}) != s[1:] ({repr(s[1:])})"
    print("✓ result[1:] == s[1:]")
except AssertionError as e:
    print(f"✗ {e}")

# Test with other examples
print("\n--- Testing other strings ---")
test_cases = [
    'hello',
    'Hello',
    'HELLO',
    '',
    'a',
    'ßeta',  # German ß at start
    'straße', # German ß at end
]

for test in test_cases:
    result = capitalize_first_letter(test)
    print(f"Input: {repr(test):12} -> Output: {repr(result):12} | Preserves rest: {result[1:] == test[1:]}")