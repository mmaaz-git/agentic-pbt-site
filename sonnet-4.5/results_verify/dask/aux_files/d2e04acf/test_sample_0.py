import tempfile
import os
from dask.bytes.core import read_bytes
from dask.utils import parse_bytes

# Test what happens with sample=0
with tempfile.TemporaryDirectory() as tmpdir:
    test_file = os.path.join(tmpdir, 'test.txt')
    with open(test_file, 'wb') as f:
        f.write(b'hello world')

    # Test different sample values
    test_cases = [
        (False, "False (no sample)"),
        (0, "0 (integer zero)"),
        (1, "1 (one byte)"),
        ("0", "string '0'"),
        ("1 B", "string '1 B'"),
    ]

    for sample_val, desc in test_cases:
        print(f"\nTesting sample={sample_val} ({desc}):")
        sample, blocks = read_bytes(test_file, sample=sample_val, blocksize=None)
        print(f"  Type: {type(sample).__name__}")
        print(f"  Value: {sample!r}")

    # Test parse_bytes on "0"
    print(f"\nparse_bytes('0') = {parse_bytes('0')}")
    print(f"parse_bytes('0 B') = {parse_bytes('0 B')}")
    print(f"parse_bytes('1 B') = {parse_bytes('1 B')}")