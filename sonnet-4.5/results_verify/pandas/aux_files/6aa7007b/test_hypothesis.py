import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, assume, example
from pandas.io import clipboards
import pandas as pd
from unittest.mock import patch

@given(st.lists(
    st.text(alphabet=st.characters(min_codepoint=32, max_codepoint=126), min_size=1, max_size=10),
    min_size=2,
    max_size=10
))
def test_tab_detection_without_trailing_newline(lines):
    assume(all('\n' not in line for line in lines))

    tab_count = 2
    lines_with_tabs = ["\t" * tab_count + line for line in lines]
    clipboard_text = "\n".join(lines_with_tabs)

    with patch('pandas.io.clipboard.clipboard_get', return_value=clipboard_text):
        with patch('pandas.io.parsers.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame()
            clipboards.read_clipboard()

            call_args = mock_read_csv.call_args
            sep_used = call_args[1].get('sep')

            parsed_lines = clipboard_text[:10000].split("\n")[:-1][:10]

            if len(parsed_lines) == 1 and len(lines) >= 2:
                assert sep_used == '\t', \
                    f"Tab detection failed. Only {len(parsed_lines)} line parsed, expected {len(lines)}"

def test_specific_example():
    lines = ["A\tB", "C\tD"]
    # Skip the assume check for direct testing
    tab_count = 2
    lines_with_tabs = ["\t" * tab_count + line for line in lines]
    clipboard_text = "\n".join(lines_with_tabs)

    with patch('pandas.io.clipboard.clipboard_get', return_value=clipboard_text):
        with patch('pandas.io.parsers.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame()
            clipboards.read_clipboard()

            call_args = mock_read_csv.call_args
            sep_used = call_args[1].get('sep')

            parsed_lines = clipboard_text[:10000].split("\n")[:-1][:10]

            if len(parsed_lines) == 1 and len(lines) >= 2:
                assert sep_used == '\t', \
                    f"Tab detection failed. Only {len(parsed_lines)} line parsed, expected {len(lines)}"

if __name__ == "__main__":
    # Test with the specific failing example
    try:
        test_specific_example()
        print("Test passed for specific example")
    except AssertionError as e:
        print(f"Test failed as expected: {e}")