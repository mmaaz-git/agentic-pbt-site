from hypothesis import given, strategies as st
from scipy.io.arff._arffread import split_data_line


@given(st.just(''))
def test_split_data_line_empty_string(line):
    result, dialect = split_data_line(line)

# Run the test
if __name__ == "__main__":
    test_split_data_line_empty_string()