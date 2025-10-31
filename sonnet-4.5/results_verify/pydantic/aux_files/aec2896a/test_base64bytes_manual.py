from pydantic import BaseModel, ValidationError
from pydantic.types import Base64Bytes

# Manual test with specific failing inputs
def test_specific_inputs():
    class Model(BaseModel):
        field: Base64Bytes

    test_cases = [
        b'\x00',
        b'\x01\x02\x03',
        b'\x80\xff\xfe',
    ]

    for data in test_cases:
        print(f"\nTesting with {data!r} ({len(data)} bytes):")
        try:
            m = Model(field=data)
            print(f"  Result: {m.field!r} ({len(m.field)} bytes)")
            if len(m.field) == 0 and len(data) > 0:
                print(f"  ❌ SILENT DATA LOSS: {len(data)} bytes became {len(m.field)} bytes")
            else:
                print(f"  ✓ Data preserved")
        except ValidationError as e:
            print(f"  ValidationError raised: {e}")

if __name__ == "__main__":
    test_specific_inputs()