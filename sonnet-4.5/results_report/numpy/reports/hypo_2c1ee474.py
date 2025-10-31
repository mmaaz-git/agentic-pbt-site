from hypothesis import assume, given, settings, strategies as st
import pandas.io.formats.format as fmt


@given(
    num=st.floats(allow_nan=False, allow_infinity=False, min_value=-1e50, max_value=1e50),
    accuracy=st.integers(min_value=1, max_value=10)
)
@settings(max_examples=500)
def test_engformatter_parse_roundtrip(num, accuracy):
    assume(num != 0)

    formatter = fmt.EngFormatter(accuracy=accuracy, use_eng_prefix=False)
    formatted = formatter(num)

    parsed = float(formatted)
    relative_error = abs(parsed - num) / abs(num) if num != 0 else 0

    max_expected_error = 10 ** (-accuracy + 1)
    assert relative_error < max_expected_error, \
        f"Round-trip error too large: {num} -> '{formatted}' -> {parsed}, error: {relative_error}"

# Run the test
if __name__ == "__main__":
    test_engformatter_parse_roundtrip()