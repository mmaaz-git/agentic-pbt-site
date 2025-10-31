import tempfile
import os
from dask.bytes.core import read_bytes

# Test case: file with 1 byte, blocksize=1, not_zero=True
with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test.txt")
    with open(filepath, 'wb') as f:
        f.write(b'a')

    print("Testing read_bytes with not_zero=True on 1-byte file with blocksize=1")
    print("=" * 60)

    sample, blocks = read_bytes(filepath, blocksize=1, not_zero=True, sample=False)

    print(f"Number of blocks: {len(blocks[0])}")

    if len(blocks[0]) > 0:
        result = blocks[0][0].compute()
        print(f"Block 0 content: {result!r}")
        print(f"Block 0 length: {len(result)}")

        if len(result) == 0:
            print("\nBUG CONFIRMED: Empty block returned!")
            print("Expected: Either no blocks (since we skip the only byte) or error")
            print("Got: Empty block with length 0")
    else:
        print("No blocks returned (which might be correct since we skip the only byte)")

print("\n" + "=" * 60)
print("Additional test cases:")
print("=" * 60)

# Test with 2-byte file
with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test2.txt")
    with open(filepath, 'wb') as f:
        f.write(b'ab')

    print("\n2-byte file with blocksize=1, not_zero=True:")
    sample, blocks = read_bytes(filepath, blocksize=1, not_zero=True, sample=False)

    for i, block in enumerate(blocks[0]):
        result = block.compute()
        print(f"  Block {i}: {result!r} (length={len(result)})")

# Test with 3-byte file
with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test3.txt")
    with open(filepath, 'wb') as f:
        f.write(b'abc')

    print("\n3-byte file with blocksize=1, not_zero=True:")
    sample, blocks = read_bytes(filepath, blocksize=1, not_zero=True, sample=False)

    for i, block in enumerate(blocks[0]):
        result = block.compute()
        print(f"  Block {i}: {result!r} (length={len(result)})")