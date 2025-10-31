#!/usr/bin/env python3
"""Test the proposed fix for the bug"""

import types

def get_version_fixed(module: types.ModuleType) -> str:
    """Fixed version of get_version"""
    version = getattr(module, "__version__", None)

    if version is None:
        raise ImportError(f"Can't determine version for {module.__name__}")
    if module.__name__ == "psycopg2":
        # psycopg2 appends " (dt dec pq3 ext lo64)" to its version
        parts = version.split()
        if not parts:
            raise ImportError(f"Can't determine version for {module.__name__}")
        version = parts[0]
    return version

print("Testing fixed version...")

# Test with whitespace-only versions for psycopg2
whitespace_strings = ["\r", " ", "\n", "\t", ""]

for ws in whitespace_strings:
    mock_module = types.ModuleType("psycopg2")
    mock_module.__version__ = ws
    try:
        result = get_version_fixed(mock_module)
        print(f"Version string {repr(ws)}: Result = {repr(result)}")
    except ImportError as e:
        print(f"Version string {repr(ws)}: ImportError - {e}")
    except Exception as e:
        print(f"Version string {repr(ws)}: {type(e).__name__} - {e}")

# Test with normal versions
print("\nTesting with normal versions...")
normal_versions = [
    "2.9.6",
    "2.9.6 (dt dec pq3 ext lo64)",
]

for ver in normal_versions:
    mock_module = types.ModuleType("psycopg2")
    mock_module.__version__ = ver
    try:
        result = get_version_fixed(mock_module)
        print(f"Version string {repr(ver)}: Result = {repr(result)}")
    except Exception as e:
        print(f"Version string {repr(ver)}: {type(e).__name__} - {e}")