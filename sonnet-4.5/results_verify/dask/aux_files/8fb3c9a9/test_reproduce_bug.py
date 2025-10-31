import tempfile
import os
from dask.bytes.core import read_bytes

print("=" * 50)
print("Testing the simple reproduction case")
print("=" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test.txt")
    with open(filepath, 'wb') as f:
        f.write(b'a')

    sample, blocks = read_bytes(filepath, blocksize=1, not_zero=True, sample=False)

    result = blocks[0][0].compute()
    print(f"Expected: non-empty block (the file has 1 byte)")
    print(f"Got: {result!r} (length={len(result)})")

print("\n" + "=" * 50)
print("Testing another edge case: file_size=2, blocksize=1")
print("=" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test.txt")
    with open(filepath, 'wb') as f:
        f.write(b'ab')

    sample, blocks = read_bytes(filepath, blocksize=1, not_zero=True, sample=False)

    print(f"Number of blocks: {len(blocks[0])}")
    for i, block in enumerate(blocks[0]):
        result = block.compute()
        print(f"Block {i}: {result!r} (length={len(result)})")

print("\n" + "=" * 50)
print("Testing file_size=3, blocksize=1")
print("=" * 50)

with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test.txt")
    with open(filepath, 'wb') as f:
        f.write(b'abc')

    sample, blocks = read_bytes(filepath, blocksize=1, not_zero=True, sample=False)

    print(f"Number of blocks: {len(blocks[0])}")
    for i, block in enumerate(blocks[0]):
        result = block.compute()
        print(f"Block {i}: {result!r} (length={len(result)})")