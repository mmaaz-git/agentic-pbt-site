#!/usr/bin/env python3
import os
import pathlib
from django.core.exceptions import SuspiciousFileOperation

def validate_file_name_original(name, allow_relative_path=False):
    """Original implementation"""
    # Remove potentially dangerous names
    if os.path.basename(name) in {"", ".", ".."}:
        raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)

    if allow_relative_path:
        # Ensure that name can be treated as a pure posix path, i.e. Unix
        # style (with forward slashes).
        path = pathlib.PurePosixPath(str(name).replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise SuspiciousFileOperation(
                "Detected path traversal attempt in '%s'" % name
            )
    elif name != os.path.basename(name):
        raise SuspiciousFileOperation("File name '%s' includes path elements" % name)

    return name

def validate_file_name_fixed(name, allow_relative_path=False):
    """Proposed fix implementation"""
    # Remove potentially dangerous names
    if os.path.basename(name) in {"", ".", ".."}:
        raise SuspiciousFileOperation("Could not derive file name from '%s'" % name)

    if allow_relative_path:
        # Ensure that name can be treated as a pure posix path, i.e. Unix
        # style (with forward slashes).
        path = pathlib.PurePosixPath(str(name).replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise SuspiciousFileOperation(
                "Detected path traversal attempt in '%s'" % name
            )
    elif name != os.path.basename(name) or "\\" in name:  # Added backslash check
        raise SuspiciousFileOperation("File name '%s' includes path elements" % name)

    return name

print("=== Testing Original vs Fixed Implementation ===")
print()

test_cases = [
    ('file\\name', False),
    ('file\\name', True),
    ('\\', False),
    ('path\\to\\file', False),
    ('dir/file', False),
    ('simple.txt', False),
    ('dir/sub\\file', False),
]

for name, allow_relative in test_cases:
    print(f"Input: {name!r}, allow_relative_path={allow_relative}")

    # Test original
    try:
        result_orig = validate_file_name_original(name, allow_relative)
        orig_status = f"Accepted: {result_orig!r}"
    except SuspiciousFileOperation as e:
        orig_status = f"Rejected: {str(e)[:50]}"

    # Test fixed
    try:
        result_fixed = validate_file_name_fixed(name, allow_relative)
        fixed_status = f"Accepted: {result_fixed!r}"
    except SuspiciousFileOperation as e:
        fixed_status = f"Rejected: {str(e)[:50]}"

    print(f"  Original: {orig_status}")
    print(f"  Fixed:    {fixed_status}")

    if orig_status != fixed_status:
        print(f"  ⚠️  BEHAVIOR CHANGED")
    print()