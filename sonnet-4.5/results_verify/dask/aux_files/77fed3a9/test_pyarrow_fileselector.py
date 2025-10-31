import pyarrow.fs as pa_fs
import tempfile
import os

# Test FileSelector with various path inputs
test_cases = [
    "/",           # Root directory
    "",            # Empty string
    ".",           # Current directory
    "/tmp",        # Valid directory
]

for path in test_cases:
    print(f"\nTesting FileSelector with base_dir='{path}':")
    try:
        selector = pa_fs.FileSelector(path, recursive=False, allow_not_found=False)
        print(f"  Created selector successfully: {selector}")
        # Try to use it with local filesystem
        fs = pa_fs.LocalFileSystem()
        try:
            file_info = fs.get_file_info(selector)
            print(f"  get_file_info returned {len(file_info)} items")
        except Exception as e:
            print(f"  get_file_info failed: {type(e).__name__}: {e}")
    except Exception as e:
        print(f"  Failed to create selector: {type(e).__name__}: {e}")