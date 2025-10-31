#!/usr/bin/env python3
"""Test the public clear_cache API"""

from scipy import datasets
import os

# Make sure test directory exists
os.makedirs("/tmp/test", exist_ok=True)

print("Testing public API (clear_cache) with invalid inputs:")
print("=" * 50)

test_inputs = [
    "not_a_callable",
    42,
    [1, 2, 3],  # not a list of callables
    ["string1", "string2"],  # list of non-callables
]

for test_input in test_inputs:
    try:
        datasets.clear_cache(test_input)
        print(f"{test_input!r}: No error raised")
    except AssertionError as e:
        print(f"{test_input!r}: AssertionError: {e}")
    except AttributeError as e:
        print(f"{test_input!r}: AttributeError: {e}")
    except Exception as e:
        print(f"{test_input!r}: {type(e).__name__}: {e}")