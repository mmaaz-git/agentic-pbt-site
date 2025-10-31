#!/usr/bin/env python3

# First, let's test the basic reproduction case
from Cython.Compiler import PyrexTypes as PT

print("Testing basic reproduction case...")
s = '000000000000000000000000000000000000000000000000000000000000000\x80'
print(f"Input string length: {len(s)}")
print(f"Input string repr: {repr(s)}")

try:
    result = PT.cap_length(s, max_len=63)
    print(f"Result: {result}")
    print("No error occurred!")
except UnicodeEncodeError as e:
    print(f"UnicodeEncodeError occurred: {e}")
    print(f"Error type: {type(e)}")
except Exception as e:
    print(f"Other error occurred: {e}")
    print(f"Error type: {type(e)}")

print("\n" + "="*50 + "\n")

# Now let's run the hypothesis test
print("Running hypothesis test...")
from hypothesis import given, strategies as st, settings

@given(st.text())
@settings(max_examples=500)
def test_cap_length_respects_max(s):
    try:
        capped = PT.cap_length(s, max_len=63)
        assert len(capped) <= 63
        return True
    except UnicodeEncodeError:
        print(f"UnicodeEncodeError on input: {repr(s)}")
        raise

# Run the test
try:
    test_cap_length_respects_max()
    print("Hypothesis test passed!")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

print("\n" + "="*50 + "\n")

# Let's also test some edge cases
print("Testing edge cases...")

# Test with ASCII-only string longer than max_len
ascii_long = "a" * 100
print(f"Testing ASCII string of length {len(ascii_long)}...")
try:
    result = PT.cap_length(ascii_long, max_len=63)
    print(f"Result length: {len(result)}")
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

print()

# Test with Unicode string shorter than max_len
unicode_short = "hello世界"
print(f"Testing Unicode string '{unicode_short}' of length {len(unicode_short)}...")
try:
    result = PT.cap_length(unicode_short, max_len=63)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

print()

# Test with Unicode string exactly at max_len
unicode_exact = "0" * 62 + "世"
print(f"Testing Unicode string of length {len(unicode_exact)} (exactly 63)...")
try:
    result = PT.cap_length(unicode_exact, max_len=63)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")

print()

# Test with Unicode string just over max_len
unicode_over = "0" * 63 + "世"
print(f"Testing Unicode string of length {len(unicode_over)} (64, over limit)...")
try:
    result = PT.cap_length(unicode_over, max_len=63)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")