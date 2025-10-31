from pydantic.v1 import BaseModel


class Model(BaseModel):
    data: bytes


m = Model(data=b'\x80')
print(f"Model created: {m}")

print("\nAttempting to serialize to JSON...")
try:
    json_str = m.json()
    print(f"Success: {json_str}")
except UnicodeDecodeError as e:
    print(f"Failed: {e}")

# Test that .dict() works fine
print("\nTesting .dict()...")
try:
    d = m.dict()
    print(f"Success: {d}")
except Exception as e:
    print(f"Failed: {e}")

# Test with UTF-8 compatible bytes
print("\nTesting with UTF-8 compatible bytes...")
m2 = Model(data=b'hello')
try:
    json_str = m2.json()
    print(f"Success: {json_str}")
except Exception as e:
    print(f"Failed: {e}")