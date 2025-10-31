#!/usr/bin/env python3
"""
Minimal reproduction of the pandas read_clipboard() bug where the last line
is lost when text doesn't end with a newline character.
"""

# Simulate the exact logic from pandas/io/clipboards.py line 98
def simulate_pandas_line_processing(text):
    """Simulates the buggy line processing from pandas read_clipboard"""
    lines = text[:10000].split("\n")[:-1][:10]
    return lines

# Test case 1: Text WITH trailing newline
text_with_newline = "a\tb\nc\td\ne\tf\n"
print("Test 1: Text WITH trailing newline")
print(f"Input: {repr(text_with_newline)}")
print(f"Split result: {text_with_newline.split('\n')}")
processed_with = simulate_pandas_line_processing(text_with_newline)
print(f"After [:-1]: {processed_with}")
print(f"Expected: ['a\\tb', 'c\\td', 'e\\tf']")
assert processed_with == ["a\tb", "c\td", "e\tf"], f"FAILED: Expected 3 lines, got {processed_with}"
print("✓ PASS: All 3 lines preserved\n")

# Test case 2: Text WITHOUT trailing newline
text_without_newline = "a\tb\nc\td\ne\tf"
print("Test 2: Text WITHOUT trailing newline")
print(f"Input: {repr(text_without_newline)}")
print(f"Split result: {text_without_newline.split('\n')}")
processed_without = simulate_pandas_line_processing(text_without_newline)
print(f"After [:-1]: {processed_without}")
print(f"Expected: ['a\\tb', 'c\\td', 'e\\tf']")
try:
    assert processed_without == ["a\tb", "c\td", "e\tf"], f"FAILED: Expected 3 lines, got {processed_without}"
    print("✓ PASS: All 3 lines preserved")
except AssertionError as e:
    print(f"✗ FAIL: {e}")
    print("BUG CONFIRMED: Last line 'e\\tf' was lost!")

# Test case 3: Single line without newline (edge case)
print("\nTest 3: Single line without newline")
single_line = "single\tline"
print(f"Input: {repr(single_line)}")
result_single = simulate_pandas_line_processing(single_line)
print(f"Result: {result_single}")
if not result_single:
    print("✗ CRITICAL BUG: Single line completely lost, result is empty!")
else:
    print(f"Lines preserved: {len(result_single)}")

# Test case 4: Regular text without tabs
print("\nTest 4: Regular text without tabs (no newline)")
regular_text = "line1\nline2\nline3\nline4"
print(f"Input: {repr(regular_text)}")
result_regular = simulate_pandas_line_processing(regular_text)
print(f"Result: {result_regular}")
print(f"Expected: ['line1', 'line2', 'line3', 'line4']")
if result_regular != ["line1", "line2", "line3", "line4"]:
    print(f"✗ FAIL: Expected 4 lines, got {len(result_regular)} lines")
    print(f"Missing: line4")

print("\n" + "="*60)
print("SUMMARY: The bug causes data loss when clipboard text doesn't end with \\n")
print("Impact: Last line is silently discarded, causing potential data corruption")