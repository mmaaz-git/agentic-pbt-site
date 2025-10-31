#!/usr/bin/env python3

# First test - the hypothesis test
print("Testing with Hypothesis:")
print("=" * 50)

from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line

@given(st.text())
def test_split_data_line_handles_all_strings(line):
    try:
        result, dialect = split_data_line(line)
        assert isinstance(result, list)
    except ValueError:
        pass

try:
    test_split_data_line_handles_all_strings()
    print("Hypothesis test passed")
except Exception as e:
    print(f"Hypothesis test failed: {e}")

print("\n" + "=" * 50)
print("Testing with empty string directly:")
print("=" * 50)

# Second test - direct reproduction
try:
    result, dialect = split_data_line('')
    print(f"Result: {result}")
    print(f"Dialect: {dialect}")
except IndexError as e:
    print(f"IndexError occurred: {e}")
except Exception as e:
    print(f"Other exception occurred: {type(e).__name__}: {e}")