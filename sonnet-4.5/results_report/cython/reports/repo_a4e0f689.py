#!/usr/bin/env python3
"""
Minimal test case demonstrating the os.path.commonprefix bug in pyximport.build_module
"""

import os
import sys

# Test case 1: Unix-style paths
path1_unix = "/tmp/very_long_directory_name_abc/subdir/file.pyx"
path2_unix = "/tmp/very_long_directory_name_abd/build"

print("=== Unix-style Paths ===")
print(f"Path 1: {path1_unix}")
print(f"Path 2: {path2_unix}")

common_unix = os.path.commonprefix([path1_unix, path2_unix])
print(f"commonprefix result: '{common_unix}'")
print(f"Is valid directory: {os.path.isdir(common_unix)}")

# Show what would happen if we tried os.chdir
print(f"\nAttempting os.chdir('{common_unix}')...")
try:
    os.chdir(common_unix)
    print("Success - changed directory")
    os.chdir("/home/npc/pbt/agentic-pbt/worker_/50")  # Change back
except FileNotFoundError as e:
    print(f"ERROR: FileNotFoundError - {e}")

# Test case 2: Windows-style paths (simulated)
print("\n=== Windows-style Paths (simulated) ===")
path1_win = "C:\\very_long_directory_name_abc\\subdir\\file.pyx"
path2_win = "C:\\very_long_directory_name_abd\\build"

print(f"Path 1: {path1_win}")
print(f"Path 2: {path2_win}")

common_win = os.path.commonprefix([path1_win, path2_win])
print(f"commonprefix result: '{common_win}'")

# Show the correct solution using commonpath
print("\n=== Correct Solution Using commonpath ===")
try:
    common_path_unix = os.path.commonpath([path1_unix, path2_unix])
    print(f"Unix paths - commonpath result: '{common_path_unix}'")
    print(f"Is valid directory: {os.path.isdir(common_path_unix)}")
except Exception as e:
    print(f"Error with commonpath: {e}")

# For Windows paths, we need to be on Windows for commonpath to work properly
# but we can show what the result would be
print("\nFor Windows paths, commonpath would return: 'C:\\'")

print("\n=== Why This Is A Bug ===")
print("os.path.commonprefix() operates character-by-character on strings,")
print("not respecting path component boundaries.")
print("This causes it to return partial directory names that don't exist.")
print("\nIn pyximport.build_module (lines 188-195), this invalid path is")
print("passed to os.chdir(), which raises FileNotFoundError and crashes.")