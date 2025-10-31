#!/usr/bin/env python3
"""
Demonstration of the bug in dask.dataframe.dask_expr.io.parquet._normalize_and_strip_protocol
This function incorrectly returns empty strings for root path inputs.
"""

from dask.dataframe.dask_expr.io.parquet import _normalize_and_strip_protocol

print("Testing _normalize_and_strip_protocol with various root path inputs:")
print("-" * 70)

# Test 1: Single forward slash (root directory)
result = _normalize_and_strip_protocol("/")
print(f"Input: '/'")
print(f"Output: {result!r}")
print(f"Result is empty string: {result == ['']}")
print()

# Test 2: Multiple forward slashes
result = _normalize_and_strip_protocol("///")
print(f"Input: '///'")
print(f"Output: {result!r}")
print(f"Result is empty string: {result == ['']}")
print()

# Test 3: Protocol with only slashes
result = _normalize_and_strip_protocol("s3:///")
print(f"Input: 's3:///'")
print(f"Output: {result!r}")
print(f"Result is empty string: {result == ['']}")
print()

# Test 4: Mixed paths including root
result = _normalize_and_strip_protocol(["s3://bucket/", "/"])
print(f"Input: ['s3://bucket/', '/']")
print(f"Output: {result!r}")
print(f"Contains empty string: {'' in result}")
print()

# Demonstrate downstream failure with PyArrow
print("-" * 70)
print("Demonstrating downstream failure with PyArrow FileSelector:")
print()

import pyarrow.fs as pa_fs
import pyarrow as pa

# This works fine with "/"
try:
    fs = pa_fs.LocalFileSystem()
    selector = pa_fs.FileSelector("/", recursive=False)
    print("FileSelector with '/' path: SUCCESS")
    # We can even list files (limiting output for brevity)
    files = fs.get_file_info(selector)[:3]
    print(f"  Found {len(fs.get_file_info(selector))} items in root directory")
except Exception as e:
    print(f"FileSelector with '/' path: FAILED - {e}")

print()

# This fails with empty string
try:
    fs = pa_fs.LocalFileSystem()
    selector = pa_fs.FileSelector("", recursive=False)
    print("FileSelector with empty string: SUCCESS")
    files = fs.get_file_info(selector)
    print(f"  Found {len(files)} items")
except Exception as e:
    print(f"FileSelector with empty string: FAILED - {e}")

print()
print("-" * 70)
print("CONCLUSION: The function transforms valid paths ('/') into invalid ones ('')")
print("This causes downstream failures in PyArrow operations.")