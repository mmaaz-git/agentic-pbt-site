#!/usr/bin/env python3
"""Test specific empty string cases mentioned in the bug report"""

from scipy.io.arff._arffread import split_data_line

# Test cases that would produce empty strings after split
test_cases = [
    ("", "Empty string"),
    ("foo\n", "String with trailing newline (split produces ['foo', ''])"),
    ("\n", "Just newline (split produces ['', ''])"),
    ("\n\n", "Double newline (split produces ['', '', ''])"),
]

print("Testing how str.split('\\n') produces empty strings:")
for test_str, description in test_cases:
    split_result = test_str.split("\n")
    print(f"\n  {repr(test_str)}.split('\\n') → {repr(split_result)}")
    print(f"  Description: {description}")

    # Test each resulting string with split_data_line
    for i, line in enumerate(split_result):
        try:
            result = split_data_line(line)
            print(f"    split_data_line({repr(line)}) → {result}")
        except IndexError as e:
            print(f"    split_data_line({repr(line)}) → IndexError: {e}")
        except Exception as e:
            print(f"    split_data_line({repr(line)}) → {type(e).__name__}: {e}")