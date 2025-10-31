import os

path1 = "/tmp/very_long_directory_name_abc/subdir/file.pyx"
path2 = "/tmp/very_long_directory_name_abd/build"

common = os.path.commonprefix([path1, path2])
print(f"Common prefix: {common}")
print(f"Is valid directory: {os.path.isdir(common)}")
print(f"Would crash on: os.chdir('{common}')")

# Show the difference between commonprefix and commonpath
print("\n--- Comparison ---")
print(f"os.path.commonprefix: {os.path.commonprefix([path1, path2])}")
print(f"os.path.commonpath: {os.path.commonpath([path1, path2])}")

# Demonstrate the actual issue
print("\n--- Windows-style paths ---")
win_path1 = "C:\\very_long_directory_name_abc\\subdir\\file.pyx"
win_path2 = "C:\\very_long_directory_name_abd\\build"
print(f"Path 1: {win_path1}")
print(f"Path 2: {win_path2}")
print(f"commonprefix result: {os.path.commonprefix([win_path1, win_path2])}")
# Note: commonpath would handle this correctly on Windows