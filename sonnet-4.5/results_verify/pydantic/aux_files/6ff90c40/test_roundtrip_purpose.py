from pydantic import BaseModel, Field
from typing import Any
import json

# Test what round_trip is actually meant for according to docs
# "dumped values should be valid as input for non-idempotent types such as Json[T]"

print("=== Understanding round_trip purpose ===")

class ModelWithJson(BaseModel):
    data: dict[str, Any]

# Create model with nested structures
model = ModelWithJson(data={"key": {"nested": "value"}, "list": [1, 2, 3]})

# Without round_trip
json_without = model.model_dump_json(round_trip=False)
print(f"Without round_trip: {json_without}")

# With round_trip
json_with = model.model_dump_json(round_trip=True)
print(f"With round_trip: {json_with}")

# Check if they're the same in this case
print(f"Are they the same? {json_without == json_with}")

# Now test the alias case more thoroughly
print("\n=== Checking if round_trip contract is clear ===")

class AliasModel(BaseModel):
    field: str = Field(alias="fieldAlias")

m = AliasModel(fieldAlias="test")

# The documentation says "dumped values should be valid as input"
# Let's check what "input" means

print("\nDifferent ways to create the model:")
print("1. Using alias (original way):")
try:
    m1 = AliasModel(fieldAlias="test")
    print(f"   Success: {m1}")
except Exception as e:
    print(f"   Failed: {e}")

print("2. Using field name:")
try:
    m2 = AliasModel(field="test")
    print(f"   Success: {m2}")
except Exception as e:
    print(f"   Failed: {e}")

print("3. Using dict with alias:")
try:
    m3 = AliasModel.model_validate({"fieldAlias": "test"})
    print(f"   Success: {m3}")
except Exception as e:
    print(f"   Failed: {e}")

print("4. Using dict with field name:")
try:
    m4 = AliasModel.model_validate({"field": "test"})
    print(f"   Success: {m4}")
except Exception as e:
    print(f"   Failed: {e}")

print("\nNow checking round_trip behavior:")
json_alias = m.model_dump_json(by_alias=True, round_trip=True)
json_field = m.model_dump_json(by_alias=False, round_trip=True)

print(f"JSON with by_alias=True: {json_alias}")
print(f"JSON with by_alias=False: {json_field}")

print("\nTrying to restore from by_alias=False with round_trip=True:")
try:
    restored = AliasModel.model_validate_json(json_field)
    print(f"Success: {restored}")
except Exception as e:
    print(f"Failed: {e}")