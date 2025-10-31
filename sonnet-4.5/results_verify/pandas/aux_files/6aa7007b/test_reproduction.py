import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.io import clipboards
import pandas as pd
from unittest.mock import patch

clipboard_text = "A\tB\nC\tD"

with patch('pandas.io.clipboard.clipboard_get', return_value=clipboard_text):
    with patch('pandas.io.parsers.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame()
        clipboards.read_clipboard()

        call_args = mock_read_csv.call_args
        sep_used = call_args[1].get('sep')

        print(f"Clipboard: {repr(clipboard_text)}")
        print(f"Separator used: {repr(sep_used)}")
        print(f"Expected: '\\t'")

        text = clipboard_text
        buggy_lines = text[:10000].split("\n")[:-1][:10]

        print(f"Lines parsed: {buggy_lines}")
        print(f"Count: {len(buggy_lines)}")
        print(f"Condition len(lines) > 1: {len(buggy_lines) > 1}")

# Additional test: what happens with trailing newline
print("\n--- Testing with trailing newline ---")
clipboard_text_with_newline = "A\tB\nC\tD\n"

with patch('pandas.io.clipboard.clipboard_get', return_value=clipboard_text_with_newline):
    with patch('pandas.io.parsers.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame()
        clipboards.read_clipboard()

        call_args = mock_read_csv.call_args
        sep_used = call_args[1].get('sep')

        print(f"Clipboard: {repr(clipboard_text_with_newline)}")
        print(f"Separator used: {repr(sep_used)}")
        print(f"Expected: '\\t'")

        text = clipboard_text_with_newline
        lines = text[:10000].split("\n")[:-1][:10]

        print(f"Lines parsed: {lines}")
        print(f"Count: {len(lines)}")
        print(f"Condition len(lines) > 1: {len(lines) > 1}")