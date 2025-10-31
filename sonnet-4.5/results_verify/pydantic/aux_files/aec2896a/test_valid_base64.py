from pydantic import BaseModel, ValidationError
from pydantic.types import Base64Bytes
import base64

class Model(BaseModel):
    data: Base64Bytes

# Test with valid base64 strings
test_cases = [
    (b'SGVsbG8=', b'Hello'),  # Valid base64
    (b'', b''),  # Empty should work
    (b'hello', None),  # Invalid base64 (not divisible by 4)
    (b'\x00', None),  # Invalid base64 byte
]

for input_val, expected in test_cases:
    print(f"\nInput: {input_val!r}")
    try:
        m = Model(data=input_val)
        print(f"Output: {m.data!r}")
        if expected is not None:
            if m.data == expected:
                print("✓ Correct decoding")
            else:
                print(f"❌ Expected {expected!r}")
    except ValidationError as e:
        print(f"ValidationError: {e}")
        if expected is None:
            print("✓ Expected error")
        else:
            print(f"❌ Unexpected error, expected {expected!r}")