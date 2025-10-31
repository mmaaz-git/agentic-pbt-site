from hypothesis import given, strategies as st, settings
import tempfile
import os
from dask.bytes.core import read_bytes

@given(
    file_size=st.integers(min_value=1, max_value=1000),
    blocksize=st.integers(min_value=1, max_value=100)
)
@settings(max_examples=100, deadline=None)
def test_read_bytes_with_not_zero_all_blocks_nonempty(file_size, blocksize):
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test.txt")
        data = b'x' * file_size

        with open(filepath, 'wb') as f:
            f.write(data)

        sample, blocks = read_bytes(filepath, blocksize=blocksize, not_zero=True, sample=False)

        total_bytes = 0
        for block in blocks[0]:
            result = block.compute()
            assert len(result) > 0, f"Empty block found! file_size={file_size}, blocksize={blocksize}"
            total_bytes += len(result)

        expected_total = file_size - 1
        assert total_bytes == expected_total, f"Total bytes mismatch: got {total_bytes}, expected {expected_total}"

# Run the test
print("Running property-based test...")
try:
    test_read_bytes_with_not_zero_all_blocks_nonempty()
    print("Test passed!")
except AssertionError as e:
    print(f"Test failed: {e}")

# Test the specific failing case mentioned
print("\nTesting the specific failing case: file_size=1, blocksize=1")
with tempfile.TemporaryDirectory() as tmpdir:
    filepath = os.path.join(tmpdir, "test.txt")
    data = b'x'

    with open(filepath, 'wb') as f:
        f.write(data)

    sample, blocks = read_bytes(filepath, blocksize=1, not_zero=True, sample=False)

    for i, block in enumerate(blocks[0]):
        result = block.compute()
        print(f"Block {i}: {result!r}, length={len(result)}")
        if len(result) == 0:
            print("Empty block found!")