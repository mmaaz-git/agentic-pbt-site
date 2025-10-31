import tempfile
import os
from dask.bytes.core import read_bytes

# Create a file with no newlines to demonstrate the bug
file_size = 10_000_000  # 10 MB file
sample_size = 100  # Request only 100 bytes sample
content = b'x' * file_size  # File with no newlines

# Create temporary file
with tempfile.NamedTemporaryFile(delete=False) as f:
    f.write(content)
    f.flush()
    temp_path = f.name

try:
    # Call read_bytes with sample and delimiter parameters
    sample, blocks = read_bytes(temp_path, blocksize=None, sample=sample_size, delimiter=b'\n')

    print(f"File size: {file_size:,} bytes")
    print(f"Requested sample size: {sample_size} bytes")
    print(f"Actual sample size: {len(sample):,} bytes")
    print(f"Sample exceeds requested size by factor of: {len(sample) / sample_size:.0f}x")
finally:
    # Clean up
    os.unlink(temp_path)