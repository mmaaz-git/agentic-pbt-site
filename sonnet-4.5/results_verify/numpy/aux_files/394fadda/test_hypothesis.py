import numpy.char as char
from hypothesis import given, strategies as st, settings
import traceback

@given(st.text(min_size=0, max_size=100))
@settings(max_examples=500, deadline=None)
def test_upper_matches_python(s):
    numpy_result = char.upper(s)
    numpy_str = str(numpy_result) if hasattr(numpy_result, 'item') else numpy_result
    python_result = s.upper()
    assert numpy_str == python_result, f"numpy: {repr(numpy_str)} != python: {repr(python_result)}"

print("Running Hypothesis test...")
try:
    test_upper_matches_python()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed with assertion error: {e}")
except Exception as e:
    print(f"Test failed with exception: {e}")
    traceback.print_exc()

# Also test specific failing case
print("\nTesting specific failing input 'ß':")
try:
    test_upper_matches_python('ß')
    print("Test passed for 'ß'")
except AssertionError as e:
    print(f"Test failed for 'ß': {e}")