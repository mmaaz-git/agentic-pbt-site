"""Test if the float division causes actual functional issues"""
import tempfile
import os
from dask.bytes import read_bytes

# Create a test file with known content
test_data = b"x" * 6356  # Same size as our failing example

with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(test_data)
    temp_path = f.name

try:
    # Read with the blocksize that triggers the bug
    sample, blocks = read_bytes(temp_path, blocksize=37)

    # Execute all blocks and concatenate
    result_data = b""
    for block_list in blocks:
        for block in block_list:
            result_data += block.compute()

    # Check if we get the same data back
    if result_data == test_data:
        print("SUCCESS: Data read correctly despite float division")
        print(f"Original size: {len(test_data)}")
        print(f"Read size: {len(result_data)}")
    else:
        print("ERROR: Data corruption detected!")
        print(f"Original size: {len(test_data)}")
        print(f"Read size: {len(result_data)}")
        print(f"First difference at byte: {next(i for i, (a, b) in enumerate(zip(test_data, result_data)) if a != b)}")

finally:
    os.unlink(temp_path)