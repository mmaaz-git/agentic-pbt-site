#!/usr/bin/env python3
"""Test the reported bug in django.utils.lorem_ipsum.words"""

from hypothesis import given, strategies as st
from django.utils.lorem_ipsum import words

# First, let's reproduce the exact issue
print("=== Reproducing the specific bug case ===")
result = words(-1, common=True)
actual_count = len(result.split())

print(f"words(-1, common=True) returned {actual_count} words")
print(f"Expected: 0 words or ValueError")
print(f"Actual: {actual_count} words")
print(f"Result: '{result}'")
print()

# Test with common=False
print("=== Testing with common=False ===")
try:
    result2 = words(-1, common=False)
    actual_count2 = len(result2.split()) if result2 else 0
    print(f"words(-1, common=False) returned {actual_count2} words")
    print(f"Result: '{result2}'")
except Exception as e:
    print(f"words(-1, common=False) raised {type(e).__name__}: {e}")
print()

# Test various negative values
print("=== Testing various negative values with common=True ===")
for count in [-1, -5, -10, -20, -100]:
    try:
        result = words(count, common=True)
        actual_count = len(result.split()) if result else 0
        print(f"words({count}, common=True) returned {actual_count} words")
    except Exception as e:
        print(f"words({count}, common=True) raised {type(e).__name__}: {e}")
print()

# Test positive edge cases
print("=== Testing positive edge cases ===")
for count in [0, 1, 5, 19, 20]:
    result = words(count, common=True)
    actual_count = len(result.split()) if result else 0
    print(f"words({count}, common=True) returned {actual_count} words")

# Now run the hypothesis test
print("\n=== Running Hypothesis test ===")
@given(st.integers(min_value=-100, max_value=-1))
def test_words_negative_count_invariant(count):
    result = words(count, common=True)
    word_count = len(result.split()) if result else 0
    assert word_count == 0, (
        f"words({count}, common=True) should return empty string or raise ValueError, "
        f"but returned {word_count} words"
    )

try:
    test_words_negative_count_invariant()
    print("Hypothesis test passed (no failures found)")
except AssertionError as e:
    print(f"Hypothesis test failed: {e}")
except Exception as e:
    print(f"Hypothesis test error: {e}")