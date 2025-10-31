import io
from hypothesis import given, strategies as st, assume, settings
from pandas.io.parsers import read_csv


@st.composite
def csv_with_thousands(draw):
    num_rows = draw(st.integers(min_value=2, max_value=10))
    rows = []
    for _ in range(num_rows):
        val = draw(st.integers(min_value=1000, max_value=999999))
        rows.append(val)
    csv_str = "number\n"
    for val in rows:
        formatted = f"{val:,}"
        csv_str += f"{formatted}\n"
    return csv_str, rows


@settings(max_examples=100)
@given(csv_with_thousands())
def test_thousands_separator(data_tuple):
    csv_str, expected_values = data_tuple
    df = read_csv(io.StringIO(csv_str), thousands=",")
    assert len(df) == len(expected_values), f"Length mismatch: got {len(df)}, expected {len(expected_values)}"
    for i, expected_val in enumerate(expected_values):
        actual_val = df.iloc[i]["number"]
        assert actual_val == expected_val, f"Row {i}: got {actual_val}, expected {expected_val}. CSV: {repr(csv_str)}"

# Run the test and catch the first failure
print("Running hypothesis test...")
try:
    test_thousands_separator()
    print("All tests passed!")
except AssertionError as e:
    print(f"Test failed: {e}")

# Test the specific failing case from the bug report
print("\n=== Testing the specific failing case ===")
csv_str, expected_values = ('number\n1,000\n1,000\n', [1000, 1000])
print(f"Input CSV: {repr(csv_str)}")
print(f"Expected values: {expected_values}")

df = read_csv(io.StringIO(csv_str), thousands=",")
print(f"Actual dataframe:\n{df}")
print(f"Actual values: {[df.iloc[i]['number'] for i in range(len(df))]}")