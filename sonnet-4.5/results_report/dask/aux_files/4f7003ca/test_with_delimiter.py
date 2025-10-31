from io import BytesIO
from fsspec.utils import read_block

# Test that length=None works with a delimiter
data = b"Hello\nWorld\nTest!"
f = BytesIO(data)

print("Testing read_block with length=None and delimiter=b'\\n'...")
try:
    result = read_block(f, 0, None, delimiter=b'\n')
    print(f"Success! Result: {result}")
except Exception as e:
    print(f"Failed: {type(e).__name__}: {e}")
