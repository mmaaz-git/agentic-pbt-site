#!/usr/bin/env python3

import traceback
from scipy.io.arff._arffread import split_data_line

print("Testing with empty string - full traceback:")
print("=" * 50)

try:
    result, dialect = split_data_line('')
except Exception as e:
    print(f"Exception type: {type(e).__name__}")
    print(f"Exception message: {e}")
    print("\nFull traceback:")
    traceback.print_exc()

print("\n" + "=" * 50)
print("Testing the exact line that causes the issue:")
print("=" * 50)

# Simulate what happens in the function
line = ''
try:
    if line[-1] == '\n':
        print("This won't be reached")
except IndexError as e:
    print(f"IndexError when accessing line[-1] on empty string: {e}")