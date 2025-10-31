from pydantic import BaseModel
from pydantic.types import Base64Bytes

class Model(BaseModel):
    data: Base64Bytes

# Test with invalid base64 bytes that cause silent data loss
test_cases = [
    b'\x00',
    b'\x01\x02\x03',
    b'\x80\xff\xfe',
]

for test_data in test_cases:
    m = Model(data=test_data)
    print(f"Input: {test_data!r} ({len(test_data)} byte{'s' if len(test_data) != 1 else ''})")
    print(f"Output: {m.data!r} ({len(m.data)} byte{'s' if len(m.data) != 1 else ''})")
    if len(test_data) > 0 and len(m.data) == 0:
        print("SILENT DATA LOSS!")
    print()