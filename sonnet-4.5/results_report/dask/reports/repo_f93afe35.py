import tempfile
from pathlib import Path

from dask.bytes.core import read_bytes

data = b'\x00\n\x00'

with tempfile.NamedTemporaryFile(delete=False, mode='wb') as f:
    f.write(data)
    temp_path = f.name

sample, blocks = read_bytes(temp_path, blocksize=None, delimiter=b'\n', sample=100)

print(f"File content: {data!r}")
print(f"Sample: {sample!r}")
print(f"Sample ends with delimiter: {sample.endswith(b'\\n')}")

Path(temp_path).unlink()