#!/usr/bin/env python3
"""Test script to reproduce the reported bug in pandas.compat._optional.get_version"""

import sys
import types
import traceback

# Import the function we're testing
from pandas.compat._optional import get_version

print("=" * 60)
print("Testing pandas.compat._optional.get_version")
print("=" * 60)

# Test 1: Property-based test with whitespace-only version
print("\n[Test 1] Creating psycopg2 module with whitespace-only version '\\r'...")
mock_module = types.ModuleType("psycopg2")
mock_module.__version__ = "\r"

try:
    result = get_version(mock_module)
    print(f"Result: {repr(result)}")
except IndexError as e:
    print(f"IndexError raised: {e}")
    print("Traceback:")
    traceback.print_exc()
except ImportError as e:
    print(f"ImportError raised: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")

# Test 2: Test with other whitespace-only strings
print("\n[Test 2] Testing with various whitespace-only strings...")
whitespace_strings = [
    " ",      # single space
    "  ",     # multiple spaces
    "\n",     # newline
    "\t",     # tab
    "\r\n",   # carriage return + newline
    "   \t\n\r  ",  # mixed whitespace
]

for ws in whitespace_strings:
    mock_module = types.ModuleType("psycopg2")
    mock_module.__version__ = ws
    try:
        result = get_version(mock_module)
        print(f"Version string {repr(ws)}: Result = {repr(result)}")
    except IndexError:
        print(f"Version string {repr(ws)}: IndexError raised")
    except ImportError as e:
        print(f"Version string {repr(ws)}: ImportError - {e}")
    except Exception as e:
        print(f"Version string {repr(ws)}: {type(e).__name__} - {e}")

# Test 3: Test with normal psycopg2 version strings
print("\n[Test 3] Testing with normal psycopg2 version strings...")
normal_versions = [
    "2.9.6",
    "2.9.6 (dt dec pq3 ext lo64)",
    "3.0.0 (dt dec pq3 ext)",
    "2.8.5 ",  # trailing space
]

for ver in normal_versions:
    mock_module = types.ModuleType("psycopg2")
    mock_module.__version__ = ver
    try:
        result = get_version(mock_module)
        print(f"Version string {repr(ver)}: Result = {repr(result)}")
    except Exception as e:
        print(f"Version string {repr(ver)}: {type(e).__name__} - {e}")

# Test 4: Test with empty string
print("\n[Test 4] Testing with empty string...")
mock_module = types.ModuleType("psycopg2")
mock_module.__version__ = ""
try:
    result = get_version(mock_module)
    print(f"Empty string: Result = {repr(result)}")
except IndexError as e:
    print(f"Empty string: IndexError - {e}")
except ImportError as e:
    print(f"Empty string: ImportError - {e}")
except Exception as e:
    print(f"Empty string: {type(e).__name__} - {e}")

# Test 5: Test with non-psycopg2 module with whitespace version
print("\n[Test 5] Testing non-psycopg2 module with whitespace version...")
mock_module = types.ModuleType("someother")
mock_module.__version__ = "\r"
try:
    result = get_version(mock_module)
    print(f"Other module with '\\r': Result = {repr(result)}")
except Exception as e:
    print(f"Other module with '\\r': {type(e).__name__} - {e}")

print("\n" + "=" * 60)
print("Test completed")
print("=" * 60)