#!/usr/bin/env python3
"""Test script to reproduce the FileBased backend bug"""

import tempfile
from django.core.mail.backends.filebased import EmailBackend

def test_reproduction():
    """Reproduce the bug as described in the report"""
    with tempfile.TemporaryDirectory() as tmpdir:
        backend = EmailBackend(file_path=tmpdir)
        result = backend.send_messages([])

        print(f"Result of send_messages([]): {result}")
        print(f"Result type: {type(result)}")
        print(f"Result is None: {result is None}")
        print(f"Result == 0: {result == 0}")

        # The bug claims it returns None instead of 0
        assert result is None, f"Expected None but got {result}"
        assert result != 0, f"Result should not be 0 but got {result}"
        print("Bug confirmed: send_messages([]) returns None instead of 0")

if __name__ == "__main__":
    test_reproduction()