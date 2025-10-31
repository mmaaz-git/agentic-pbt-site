from pydantic import BaseModel
from pydantic.types import Base64Bytes

class Model(BaseModel):
    data: Base64Bytes

m = Model(data=b'\x00')
print(f"Input: b'\\x00' (1 byte)")
print(f"Output: {m.data!r} ({len(m.data)} bytes)")
if len(m.data) == 0:
    print(f"Silent data loss!")