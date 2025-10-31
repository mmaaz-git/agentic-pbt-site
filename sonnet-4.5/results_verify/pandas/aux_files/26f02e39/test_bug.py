from hypothesis import given, strategies as st, assume
from unittest.mock import patch
import pandas as pd

# First, let's run the basic reproduction case
print("=== Basic Reproduction Case ===")
text = " \ta\tb\tc\n \t1\t2\t3\n \t4\t5\t6\n"
print(f"Input text representation: {repr(text)}")

with patch('pandas.io.clipboard.clipboard_get', return_value=text):
    df = pd.read_clipboard()
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Index: {df.index.tolist()}")
    print("DataFrame:")
    print(df)
    print()

# Now let's run the property-based test
print("=== Property-Based Test ===")

@given(st.integers(min_value=1, max_value=5))
def test_index_column_detection_counts_tabs_not_characters(num_spaces):
    assume(num_spaces > 0)

    data_rows = [['a', 'b', 'c'], ['1', '2', '3'], ['4', '5', '6']]

    leading_ws = ' ' * num_spaces + '\t'
    lines = [leading_ws + '\t'.join(row) for row in data_rows]
    text = '\n'.join(lines) + '\n'

    print(f"Testing with {num_spaces} spaces + 1 tab")
    print(f"Leading whitespace: {repr(leading_ws)}")
    print(f"First line: {repr(lines[0])}")

    with patch('pandas.io.clipboard.clipboard_get', return_value=text):
        df = pd.read_clipboard()

        expected_tab_count = 1
        actual_column_count = df.shape[1]

        expected_columns = 3
        print(f"  Expected {expected_columns} columns, got {actual_column_count}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Shape: {df.shape}")

        assert actual_column_count == expected_columns, \
            f"With {num_spaces} spaces + 1 tab, expected {expected_columns} columns but got {actual_column_count}"

# Run the property test
try:
    test_index_column_detection_counts_tabs_not_characters()
    print("Property test PASSED (no assertion errors)")
except AssertionError as e:
    print(f"Property test FAILED: {e}")
except Exception as e:
    print(f"Property test ERROR: {e}")

# Let's also manually test a few specific cases
print("\n=== Manual Tests ===")

test_cases = [
    (" \ta\tb\tc", "1 space + 1 tab"),
    ("  \ta\tb\tc", "2 spaces + 1 tab"),
    ("\ta\tb\tc", "1 tab only"),
    ("\t\ta\tb\tc", "2 tabs only"),
    ("   \ta\tb\tc", "3 spaces + 1 tab"),
]

for first_line, description in test_cases:
    text = f"{first_line}\n{first_line.replace('a', '1').replace('b', '2').replace('c', '3')}\n"
    print(f"\nTest: {description}")
    print(f"First line: {repr(first_line)}")

    # Calculate what the code does
    leading_ws_len = len(first_line) - len(first_line.lstrip(" \t"))
    print(f"  Leading whitespace length: {leading_ws_len}")
    print(f"  Number of tabs in leading WS: {first_line[:leading_ws_len].count('\t')}")

    with patch('pandas.io.clipboard.clipboard_get', return_value=text):
        df = pd.read_clipboard()
        print(f"  Result columns: {list(df.columns)}")
        print(f"  Result shape: {df.shape}")