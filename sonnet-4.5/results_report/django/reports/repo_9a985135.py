#!/usr/bin/env python3
"""Minimal reproduction of the FileBased email backend bug."""

import tempfile
from django.core.mail.backends.filebased import EmailBackend

# Create a temporary directory for the email files
with tempfile.TemporaryDirectory() as tmpdir:
    # Initialize the FileBased email backend
    backend = EmailBackend(file_path=tmpdir, fail_silently=True)

    # Call send_messages with an empty list
    result = backend.send_messages([])

    # Print the results
    print(f"Result from send_messages([]): {result}")
    print(f"Result type: {type(result)}")
    print(f"Result is None: {result is None}")
    print(f"Result equals 0: {result == 0}")

    # This demonstrates the bug - should return 0, but returns None
    if result is None:
        print("\nBUG CONFIRMED: send_messages([]) returned None instead of 0")
    else:
        print("\nNo bug: send_messages([]) correctly returned:", result)