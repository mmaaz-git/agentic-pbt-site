#!/usr/bin/env python3
"""Test what happens when we pass empty strings to pyarrow FileSelector"""

import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/dask_env/lib/python3.13/site-packages')

import pyarrow as pa
import pyarrow.fs as pa_fs

print("Testing pyarrow.fs.FileSelector with various paths...")

# Test 1: Normal path
print("\n1. Testing with normal path '.':")
try:
    fs = pa_fs.LocalFileSystem()
    selector = pa_fs.FileSelector(".", recursive=False)
    files = fs.get_file_info(selector)
    print(f"   Success: Found {len(files)} items")
except Exception as e:
    print(f"   Error: {e}")

# Test 2: Root path '/'
print("\n2. Testing with root path '/':")
try:
    fs = pa_fs.LocalFileSystem()
    selector = pa_fs.FileSelector("/", recursive=False)
    files = fs.get_file_info(selector)
    print(f"   Success: Found {len(files)} items")
except Exception as e:
    print(f"   Error: {e}")

# Test 3: Empty string ''
print("\n3. Testing with empty string '':")
try:
    fs = pa_fs.LocalFileSystem()
    selector = pa_fs.FileSelector("", recursive=False)
    files = fs.get_file_info(selector)
    print(f"   Success: Found {len(files)} items")
except Exception as e:
    print(f"   Error: {e}")

# Test 4: fs.get_file_info with empty string directly
print("\n4. Testing fs.get_file_info with empty string directly:")
try:
    fs = pa_fs.LocalFileSystem()
    file_info = fs.get_file_info("")
    print(f"   Success: File info type: {file_info.type}")
except Exception as e:
    print(f"   Error: {e}")