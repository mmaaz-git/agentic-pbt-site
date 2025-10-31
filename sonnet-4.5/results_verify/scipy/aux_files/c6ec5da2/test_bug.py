#!/usr/bin/env python3
"""Test the reported bug in scipy.datasets._clear_cache"""

import sys
import os

# Test 1: Reproduce the basic bug without -O flag
print("Test 1: Running without -O flag")
print("=" * 50)
try:
    from scipy.datasets._utils import _clear_cache
    _clear_cache("not_a_callable", cache_dir="/tmp/test")
    print("No error raised - BUG NOT REPRODUCED")
except AssertionError as e:
    print(f"AssertionError raised as expected: {e}")
except AttributeError as e:
    print(f"AttributeError raised (unexpected without -O): {e}")
except Exception as e:
    print(f"Other exception: {type(e).__name__}: {e}")

print("\n")

# Test 2: Test with various non-callable inputs
print("Test 2: Testing with various non-callable inputs")
print("=" * 50)
test_inputs = [
    42,
    "string",
    [1, 2, 3],
    {"key": "value"},
    None,
    True,
    3.14
]

for test_input in test_inputs:
    try:
        _clear_cache(test_input, cache_dir="/tmp/test")
        print(f"{test_input!r}: No error raised - UNEXPECTED")
    except AssertionError:
        print(f"{test_input!r}: AssertionError raised")
    except AttributeError as e:
        print(f"{test_input!r}: AttributeError: {e}")
    except Exception as e:
        print(f"{test_input!r}: {type(e).__name__}: {e}")