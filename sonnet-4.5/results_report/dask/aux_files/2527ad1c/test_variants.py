import tempfile
import os
from dask.bytes.core import read_bytes

with tempfile.TemporaryDirectory() as tmpdir:
    test_file = os.path.join(tmpdir, 'test.txt')
    with open(test_file, 'wb') as f:
        f.write(b'hello world')

    # Test different sample values
    test_cases = [
        (False, "sample=False"),
        (0, "sample=0 (int)"),
        (1, "sample=1 (int)"),
        ("0", "sample='0' (str)"),
        ("1 B", "sample='1 B' (str)"),
    ]

    for sample_val, description in test_cases:
        sample, blocks = read_bytes(test_file, sample=sample_val, blocksize=None)
        print(f"{description}:")
        print(f"  Type: {type(sample).__name__}")
        print(f"  Value: {sample!r}")
        print()