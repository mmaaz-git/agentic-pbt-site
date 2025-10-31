#!/usr/bin/env python3
"""Simple reproduction of the bug"""

from scipy.io.arff._arffread import split_data_line

print("Testing split_data_line with empty string...")
try:
    result = split_data_line("")
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError caught: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")