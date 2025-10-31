# Bug Report: WSL Clipboard Paste Data Corruption

**Target**: `pandas.io.clipboard.init_wsl_clipboard().paste_wsl`
**Severity**: High
**Bug Type**: Logic
**Date**: 2025-09-25

## Summary

The WSL clipboard `paste_wsl()` function unconditionally removes the last 2 bytes from PowerShell output, assuming it always ends with `\r\n`. This causes silent data corruption when the output doesn't end with `\r\n`, and can cause `UnicodeDecodeError` crashes when the slicing splits multi-byte UTF-8 characters.

## Property-Based Test

```python
import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pandas_env')

from hypothesis import given, strategies as st, settings, assume
from unittest.mock import Mock, patch
from pandas.io.clipboard import init_wsl_clipboard


@given(st.text(min_size=1, max_size=100))
@settings(max_examples=200)
def test_wsl_paste_without_crlf_corrupts_data(text):
    assume(len(text) >= 2)

    copy_wsl, paste_wsl = init_wsl_clipboard()

    with patch('pandas.io.clipboard.subprocess.Popen') as mock_popen:
        mock_process = Mock()
        mock_process.communicate.return_value = (text.encode('utf-8'), b'')
        mock_popen.return_value.__enter__.return_value = mock_process

        result = paste_wsl()
        assert result == text, f"Data corruption: {text!r} became {result!r}"
```

**Failing inputs**:
- `text='00'` - becomes empty string `''`
- `text='0ࠀ'` - causes `UnicodeDecodeError: 'utf-8' codec can't decode byte 0xe0 in position 1: unexpected end of data`

## Reproducing the Bug

```python
from unittest.mock import Mock, patch
from pandas.io.clipboard import init_wsl_clipboard

copy_wsl, paste_wsl = init_wsl_clipboard()

print("Bug 1: Data corruption")
with patch('pandas.io.clipboard.subprocess.Popen') as mock_popen:
    mock_process = Mock()
    mock_process.communicate.return_value = (b'hello', b'')
    mock_popen.return_value.__enter__.return_value = mock_process
    result = paste_wsl()
    print(f"Input: 'hello', Output: {result!r}")

print("\nBug 2: Unicode decode error")
with patch('pandas.io.clipboard.subprocess.Popen') as mock_popen:
    mock_process = Mock()
    text = '0ࠀ'
    mock_process.communicate.return_value = (text.encode('utf-8'), b'')
    mock_popen.return_value.__enter__.return_value = mock_process
    try:
        result = paste_wsl()
    except UnicodeDecodeError as e:
        print(f"Crash: {e}")
```

## Why This Is A Bug

The code at line 520 unconditionally slices the last 2 bytes:

```python
return stdout[:-2].decode(ENCODING)
```

This assumes PowerShell's `Get-Clipboard` always appends `\r\n`. However:
1. This assumption may not always hold across different PowerShell versions or configurations
2. Even if it does, slicing bytes before decoding can split multi-byte UTF-8 characters, causing decode errors
3. For short clipboard content, this removes actual user data

## Fix

```diff
diff --git a/pandas/io/clipboard/__init__.py b/pandas/io/clipboard/__init__.py
index 1234567..abcdefg 100644
--- a/pandas/io/clipboard/__init__.py
+++ b/pandas/io/clipboard/__init__.py
@@ -517,7 +517,10 @@ def init_wsl_clipboard():
         ) as p:
             stdout = p.communicate()[0]
-        # WSL appends "\r\n" to the contents.
-        return stdout[:-2].decode(ENCODING)
+        # WSL's PowerShell may append "\r\n" to the contents.
+        # Decode first, then strip to avoid splitting multi-byte characters.
+        result = stdout.decode(ENCODING)
+        if result.endswith('\r\n'):
+            result = result[:-2]
+        return result

     return copy_wsl, paste_wsl
```