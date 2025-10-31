from pydantic.v1 import BaseModel, Field


class ModelWithDefaults(BaseModel):
    required: str
    regular_default: int = 42
    factory_default: list = Field(default_factory=list)


model = ModelWithDefaults(required='test')

d = model.dict(exclude_defaults=True)

print(f"Dictionary with exclude_defaults=True: {d}")
print(f"'regular_default' in dict: {'regular_default' in d}")
print(f"'factory_default' in dict: {'factory_default' in d}")

try:
    assert 'regular_default' not in d
    print("✓ regular_default correctly excluded")
except AssertionError:
    print("✗ regular_default NOT excluded (unexpected)")

try:
    assert 'factory_default' not in d
    print("✓ factory_default correctly excluded")
except AssertionError:
    print("✗ factory_default NOT excluded (this is the bug)")

# Let's also check the values to confirm they are defaults
print(f"\nActual values:")
print(f"model.regular_default = {model.regular_default}")
print(f"model.factory_default = {model.factory_default}")