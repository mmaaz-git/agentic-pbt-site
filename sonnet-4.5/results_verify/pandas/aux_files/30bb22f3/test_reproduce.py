import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from unittest.mock import Mock, patch
from pandas.io.clipboard import init_klipper_clipboard

copy_klipper, paste_klipper = init_klipper_clipboard()

print("Bug 1: Empty clipboard crashes")
with patch('pandas.io.clipboard.subprocess.Popen') as mock_popen:
    mock_process = Mock()
    mock_process.communicate.return_value = (b'', None)
    mock_popen.return_value.__enter__.return_value = mock_process

    try:
        result = paste_klipper()
        print(f"Result: {result!r}")
    except AssertionError as e:
        print(f"AssertionError: {e}")

print("\nBug 2: Missing trailing newline crashes")
with patch('pandas.io.clipboard.subprocess.Popen') as mock_popen:
    mock_process = Mock()
    mock_process.communicate.return_value = (b'hello', None)
    mock_popen.return_value.__enter__.return_value = mock_process

    try:
        result = paste_klipper()
        print(f"Result: {result!r}")
    except AssertionError as e:
        print(f"AssertionError at line 279")