#!/usr/bin/env python3
"""
Test the impact of having '.' in extensions set
"""
import os
import tempfile

# Simulate what makemessages does
extensions = {'.py', '.', '.js'}  # Including the problematic '.'

print("Testing impact of '.' in extensions set:")
print("=" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    # Create test files
    files = [
        "test.py",
        "test.js",
        "test.txt",
        "test",       # No extension
        ".hidden",    # Hidden file
        "file.",      # Trailing dot
    ]

    for fname in files:
        open(os.path.join(tmpdir, fname), 'w').close()

    print("Files created:")
    for fname in files:
        print(f"  - {fname}")

    print("\nChecking which files would match extensions {'.py', '.', '.js'}:")
    for fname in files:
        file_ext = os.path.splitext(fname)[1]
        matches = file_ext in extensions
        print(f"  {fname:15} -> ext='{file_ext:5}' -> matches={matches}")

print("\nPotential issues:")
print("  - 'file.' has extension '.' and would match unexpectedly")
print("  - This could cause unwanted files to be processed")
print("  - While 'file.' is uncommon, it can exist in filesystems")