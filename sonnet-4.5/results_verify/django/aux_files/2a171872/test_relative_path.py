#!/usr/bin/env python3
import pathlib

print("=== Understanding allow_relative_path=True behavior ===")
print()

test_cases = [
    'file\\name',
    'folder\\file.txt',
    'path\\to\\file.txt',
    '../parent/file.txt',
    '..\\parent\\file.txt',
    '/absolute/path',
    'C:\\Windows\\path',
    'simple.txt',
]

for name in test_cases:
    print(f"Input: {name!r}")

    # What the code does in allow_relative_path=True
    converted = str(name).replace("\\", "/")
    print(f"  After replace('\\\\', '/'): {converted!r}")

    path = pathlib.PurePosixPath(converted)
    print(f"  PurePosixPath parts: {path.parts}")
    print(f"  Is absolute: {path.is_absolute()}")
    print(f"  Contains '..': {'..' in path.parts}")

    would_reject = path.is_absolute() or ".." in path.parts
    print(f"  Would be rejected: {would_reject}")
    print()