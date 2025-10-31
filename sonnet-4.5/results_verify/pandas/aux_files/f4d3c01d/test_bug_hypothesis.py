from hypothesis import given, strategies as st
from pandas.io.formats.format import _trim_zeros_complex


@given(st.lists(st.tuples(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False)
), min_size=1, max_size=10))
def test_trim_zeros_complex_preserves_parentheses(float_pairs):
    values = [complex(r, i) for r, i in float_pairs]
    str_complexes = [str(v) for v in values]
    trimmed = _trim_zeros_complex(str_complexes)

    for original, result in zip(str_complexes, trimmed):
        if original.endswith(')'):
            assert result.endswith(')'), f"Lost closing parenthesis: {original} -> {result}"

# Run the test
print("Running hypothesis test...")
try:
    test_trim_zeros_complex_preserves_parentheses()
    print("Test passed (no issues found)")
except AssertionError as e:
    print(f"Test failed: {e}")
except Exception as e:
    print(f"Test error: {e}")

# Also test the specific failing input
print("\nTesting specific failing input [(1.0, 1.0)]:")
float_pairs = [(1.0, 1.0)]
values = [complex(r, i) for r, i in float_pairs]
str_complexes = [str(v) for v in values]
print(f"Input: {str_complexes}")
trimmed = _trim_zeros_complex(str_complexes)
print(f"Output: {trimmed}")
if str_complexes[0].endswith(')') and not trimmed[0].endswith(')'):
    print(f"ERROR: Lost closing parenthesis!")