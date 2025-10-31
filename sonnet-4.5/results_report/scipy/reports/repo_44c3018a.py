#!/usr/bin/env python3
"""Minimal reproduction of scipy.io.arff.split_data_line IndexError on empty string."""

from scipy.io.arff._arffread import split_data_line

# Attempting to split an empty string
print("Testing split_data_line with empty string:")
try:
    result, dialect = split_data_line('')
    print(f"Success: result={result}, dialect={dialect}")
except IndexError as e:
    print(f"IndexError caught: {e}")
    import traceback
    traceback.print_exc()