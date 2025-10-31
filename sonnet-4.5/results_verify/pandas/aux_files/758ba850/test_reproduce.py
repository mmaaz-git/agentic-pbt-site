#!/usr/bin/env python3
"""Test to reproduce the Range function bug with odd-length strings"""

import sys
import traceback

# First, test the basic reproduction
print("=" * 60)
print("TEST 1: Basic reproduction with 'abc'")
print("=" * 60)
try:
    from Cython.Plex.Regexps import Range
    result = Range('abc')
    print(f"Unexpectedly succeeded: {result}")
except IndexError as e:
    print(f"Got IndexError as reported: {e}")
    traceback.print_exc()
except Exception as e:
    print(f"Got different error: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("TEST 2: Test with '000' (another odd-length string)")
print("=" * 60)
try:
    from Cython.Plex.Regexps import Range
    result = Range('000')
    print(f"Unexpectedly succeeded: {result}")
except IndexError as e:
    print(f"Got IndexError as reported: {e}")
except Exception as e:
    print(f"Got different error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("TEST 3: Test with single character 'a'")
print("=" * 60)
try:
    from Cython.Plex.Regexps import Range
    result = Range('a')
    print(f"Unexpectedly succeeded: {result}")
except IndexError as e:
    print(f"Got IndexError as reported: {e}")
except Exception as e:
    print(f"Got different error: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("TEST 4: Test with even-length string 'abcd' (should work)")
print("=" * 60)
try:
    from Cython.Plex.Regexps import Range
    result = Range('abcd')
    print(f"Success with even-length string: {result}")
    print(f"Result string representation: {result.str}")
except Exception as e:
    print(f"Unexpected error with even-length string: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("TEST 5: Test with two-argument form (should work)")
print("=" * 60)
try:
    from Cython.Plex.Regexps import Range
    result = Range('a', 'z')
    print(f"Success with two arguments: {result}")
    print(f"Result string representation: {result.str}")
except Exception as e:
    print(f"Unexpected error with two arguments: {type(e).__name__}: {e}")

print("\n" + "=" * 60)
print("TEST 6: Property-based test with hypothesis")
print("=" * 60)
try:
    from hypothesis import given, strategies as st, settings
    from Cython.Plex.Regexps import Range

    errors_found = []

    @given(st.text(min_size=1).filter(lambda s: len(s) % 2 == 1))
    @settings(max_examples=10)  # Reduced for quick testing
    def test_range_validates_even_length(s):
        try:
            Range(s)
            errors_found.append(f"No error for odd-length string: {repr(s)}")
        except IndexError:
            pass  # Expected
        except Exception as e:
            errors_found.append(f"Unexpected error for {repr(s)}: {e}")

    test_range_validates_even_length()

    if errors_found:
        print("Errors found in property-based test:")
        for error in errors_found:
            print(f"  - {error}")
    else:
        print("Property-based test confirmed: All odd-length strings raise IndexError")

except ImportError:
    print("Hypothesis not installed, skipping property-based test")
except Exception as e:
    print(f"Error in property-based test: {e}")
    traceback.print_exc()