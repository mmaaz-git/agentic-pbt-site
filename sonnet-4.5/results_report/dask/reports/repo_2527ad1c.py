import tempfile
import os
from dask.bytes.core import read_bytes

with tempfile.TemporaryDirectory() as tmpdir:
    test_file = os.path.join(tmpdir, 'test.txt')
    with open(test_file, 'wb') as f:
        f.write(b'hello world')

    sample, blocks = read_bytes(test_file, sample=0, blocksize=None)

    print(f"Type: {type(sample)}")
    print(f"Value: {sample!r}")

    # This will fail with AttributeError since sample is int, not bytes
    try:
        sample.decode('utf-8')
        print("Successfully decoded as UTF-8")
    except AttributeError as e:
        print(f"AttributeError: {e}")