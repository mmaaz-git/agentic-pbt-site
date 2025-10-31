import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from pandas.io import clipboards
import pandas as pd
from unittest.mock import patch

# Test case 1: Without trailing newline (should fail)
print("=" * 60)
print("TEST CASE 1: Clipboard text WITHOUT trailing newline")
print("=" * 60)

clipboard_text = "A\tB\nC\tD"

with patch('pandas.io.clipboard.clipboard_get', return_value=clipboard_text):
    with patch('pandas.io.parsers.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame()
        clipboards.read_clipboard()

        call_args = mock_read_csv.call_args
        sep_used = call_args[1].get('sep')

        print(f"Clipboard text: {repr(clipboard_text)}")
        print(f"Separator detected: {repr(sep_used)}")
        print(f"Expected separator: '\\t'")
        print(f"Detection correct: {sep_used == '\\t'}")

        # Show what lines are parsed
        text = clipboard_text
        buggy_lines = text[:10000].split("\n")[:-1][:10]

        print(f"\nLines parsed by buggy code: {buggy_lines}")
        print(f"Number of lines parsed: {len(buggy_lines)}")
        print(f"Condition 'len(lines) > 1': {len(buggy_lines) > 1}")

print("\n" + "=" * 60)
print("TEST CASE 2: Clipboard text WITH trailing newline")
print("=" * 60)

# Test case 2: With trailing newline (should work)
clipboard_text_with_newline = "A\tB\nC\tD\n"

with patch('pandas.io.clipboard.clipboard_get', return_value=clipboard_text_with_newline):
    with patch('pandas.io.parsers.read_csv') as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame()
        clipboards.read_clipboard()

        call_args = mock_read_csv.call_args
        sep_used = call_args[1].get('sep')

        print(f"Clipboard text: {repr(clipboard_text_with_newline)}")
        print(f"Separator detected: {repr(sep_used)}")
        print(f"Expected separator: '\\t'")
        print(f"Detection correct: {sep_used == '\\t'}")

        # Show what lines are parsed
        text = clipboard_text_with_newline
        buggy_lines = text[:10000].split("\n")[:-1][:10]

        print(f"\nLines parsed by buggy code: {buggy_lines}")
        print(f"Number of lines parsed: {len(buggy_lines)}")
        print(f"Condition 'len(lines) > 1': {len(buggy_lines) > 1}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("The bug occurs when clipboard text doesn't end with a newline.")
print("The code unconditionally removes the last line with [:-1],")
print("which discards valid data when there's no trailing newline.")