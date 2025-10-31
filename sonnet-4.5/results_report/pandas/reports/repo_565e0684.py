#!/usr/bin/env python3
"""
Minimal reproduction of pandas Klipper clipboard bug with null bytes.
"""
import subprocess

ENCODING = 'utf-8'
text = "hello\x00world"

try:
    with subprocess.Popen(
        [
            "qdbus",
            "org.kde.klipper",
            "/klipper",
            "setClipboardContents",
            text.encode(ENCODING),  # This is the bug - passing bytes instead of string
        ],
        stdin=subprocess.PIPE,
        close_fds=True,
    ) as p:
        p.communicate(input=None)
    print("No error occurred")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")