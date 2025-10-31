# Bug Report: Flask send_from_directory Crashes on Filenames with Newline Characters

**Target**: `flask.helpers.send_from_directory`
**Severity**: Medium
**Bug Type**: Crash
**Date**: 2025-09-25

## Summary

`send_from_directory` crashes with a `ValueError` when attempting to serve files whose names contain newline characters (`\r` or `\n`), even though such files can exist on Unix filesystems and pass all security checks.

## Property-Based Test

```python
from flask import Flask
from flask.helpers import send_from_directory
from hypothesis import given, strategies as st
import tempfile
import os

app = Flask(__name__)

@given(st.text(alphabet=st.characters(blacklist_characters='/\\\x00:*?"<>|'), min_size=1, max_size=20))
def test_nested_directory_access(filename):
    """Property: Files with OS-valid filenames should be servable"""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, filename)
        with open(filepath, 'w') as f:
            f.write("content")

        with app.test_request_context():
            response = send_from_directory(tmpdir, filename)
            assert response.status_code == 200
```

**Failing input**: `filename='\r'`

## Reproducing the Bug

```python
from flask import Flask
from flask.helpers import send_from_directory
import tempfile
import os

app = Flask(__name__)

with tempfile.TemporaryDirectory() as tmpdir:
    filename = "test\rfile.txt"
    filepath = os.path.join(tmpdir, filename)

    with open(filepath, 'w') as f:
        f.write("content")

    with app.test_request_context():
        response = send_from_directory(tmpdir, filename)
```

Output:
```
ValueError: Header values must not contain newline characters.
```

## Why This Is A Bug

1. **Undocumented limitation**: The documentation claims `send_from_directory` is a "secure way to serve files" but doesn't mention it will crash on certain OS-valid filenames
2. **Security implication**: Malicious users could upload files with newlines in their names to cause a DoS
3. **Inconsistent behavior**: The function's security checks (`safe_join`) pass for these filenames, but it later crashes when setting HTTP headers
4. **Poor error message**: The error doesn't explain that the issue is with the filename

## Fix

The function should sanitize filenames before using them in HTTP headers. A simple fix would be to replace newline characters in the filename used for the Content-Disposition header:

```diff
diff --git a/src/flask/helpers.py b/src/flask/helpers.py
index abc1234..def5678 100644
--- a/src/flask/helpers.py
+++ b/src/flask/helpers.py
@@ -560,9 +560,13 @@ def send_from_directory(
         raise

     if isinstance(path, os.PathLike):
         path = os.fspath(path)

+    # Sanitize filename for HTTP headers
+    if "download_name" not in kwargs and os.path.basename(path):
+        kwargs["download_name"] = os.path.basename(path).replace('\r', '').replace('\n', '')
+
     return werkzeug.utils.send_from_directory(  # type: ignore[return-value]
         directory, path, **kwargs
     )
```

Note: The actual fix should be in Werkzeug's `send_file` function, as that's where the Content-Disposition header is set. Flask could work around this by sanitizing the download_name parameter before passing it to Werkzeug.