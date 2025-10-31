from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line

@given(st.text(max_size=100))
def test_split_data_line_handles_all_strings(line):
    try:
        result, dialect = split_data_line(line)
        assert isinstance(result, list)
    except (IndexError, ValueError) as e:
        raise AssertionError(f"split_data_line crashed on input {line!r}: {e}")

# Run the test
if __name__ == "__main__":
    print("Running hypothesis test...")
    test_split_data_line_handles_all_strings()