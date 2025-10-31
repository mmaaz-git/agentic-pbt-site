from pydantic import BaseModel

class TestModel(BaseModel):
    temperature: float = 0.5
    max_tokens: int = None

model = TestModel()

print("Iterating over Pydantic model:")
try:
    for item in model:
        print(f"  Item: {item}, Type: {type(item)}")
except Exception as e:
    print(f"  Error: {e}")

print("\nIterating over regular dict:")
d = {'temperature': 0.5, 'max_tokens': None}
for item in d:
    print(f"  Item: {item}, Type: {type(item)}")

print("\nTrying dict comprehension on Pydantic model:")
try:
    result = {key: value for key, value in model if value is not None}
    print(f"  Success: {result}")
except Exception as e:
    print(f"  Error: {e}")

print("\nTrying dict comprehension on regular dict:")
try:
    result = {key: value for key, value in d if value is not None}
    print(f"  Success: {result}")
except Exception as e:
    print(f"  Error: {e}")