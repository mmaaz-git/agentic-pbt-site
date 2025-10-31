from hypothesis import given, strategies as st, settings
import numpy as np
import numpy.char as nc

# Test upper
@given(st.text(min_size=1))
@settings(max_examples=100)
def test_upper_matches_python(s):
    arr = np.array([s])
    numpy_result = nc.upper(arr)[0]
    python_result = s.upper()
    assert numpy_result == python_result, f"Failed for {s!r}: numpy={numpy_result!r}, python={python_result!r}"

# Test lower
@given(st.text(min_size=1))
@settings(max_examples=100)
def test_lower_matches_python(s):
    arr = np.array([s])
    numpy_result = nc.lower(arr)[0]
    python_result = s.lower()
    assert numpy_result == python_result, f"Failed for {s!r}: numpy={numpy_result!r}, python={python_result!r}"

# Test swapcase
@given(st.text(min_size=1))
@settings(max_examples=100)
def test_swapcase_matches_python(s):
    arr = np.array([s])
    numpy_result = nc.swapcase(arr)[0]
    python_result = s.swapcase()
    assert numpy_result == python_result, f"Failed for {s!r}: numpy={numpy_result!r}, python={python_result!r}"

print("Running Hypothesis tests...")
print("=" * 60)

try:
    print("\nTesting upper()...")
    test_upper_matches_python()
    print("upper() test passed!")
except AssertionError as e:
    print(f"upper() test FAILED: {e}")

try:
    print("\nTesting lower()...")
    test_lower_matches_python()
    print("lower() test passed!")
except AssertionError as e:
    print(f"lower() test FAILED: {e}")

try:
    print("\nTesting swapcase()...")
    test_swapcase_matches_python()
    print("swapcase() test passed!")
except AssertionError as e:
    print(f"swapcase() test FAILED: {e}")

# Specific test for known failing cases
print("\n" + "=" * 60)
print("Testing specific failing cases:")

failing_cases = ['ß', 'ﬁ', 'ﬂ', 'ﬀ', 'ﬃ', 'ﬄ', 'ﬅ', 'ﬆ', 'İ']
for char in failing_cases:
    arr = np.array([char])

    numpy_upper = nc.upper(arr)[0]
    python_upper = char.upper()

    numpy_lower = nc.lower(arr)[0]
    python_lower = char.lower()

    numpy_swapcase = nc.swapcase(arr)[0]
    python_swapcase = char.swapcase()

    if numpy_upper != python_upper:
        print(f"✗ upper('{char}'): numpy={numpy_upper!r}, python={python_upper!r}")

    if numpy_lower != python_lower:
        print(f"✗ lower('{char}'): numpy={numpy_lower!r}, python={python_lower!r}")

    if numpy_swapcase != python_swapcase:
        print(f"✗ swapcase('{char}'): numpy={numpy_swapcase!r}, python={python_swapcase!r}")