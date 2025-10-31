from pydantic import BaseModel, Field
import traceback

class ModelWithAlias(BaseModel):
    field_one: str = Field(alias="fieldOne")

model = ModelWithAlias(fieldOne="test")

print(f"Original model: {model}")
print(f"Model field_one value: {model.field_one}")

json_str = model.model_dump_json(by_alias=False, round_trip=True)
print(f"JSON with by_alias=False, round_trip=True: {json_str}")

try:
    restored = ModelWithAlias.model_validate_json(json_str)
    print(f"Restored model: {restored}")
except Exception as e:
    print(f"Error restoring model: {e}")
    traceback.print_exc()

print("\n--- Testing with by_alias=True ---")
json_str_alias = model.model_dump_json(by_alias=True, round_trip=True)
print(f"JSON with by_alias=True, round_trip=True: {json_str_alias}")

try:
    restored_alias = ModelWithAlias.model_validate_json(json_str_alias)
    print(f"Restored model: {restored_alias}")
    print(f"Restored field_one value: {restored_alias.field_one}")
except Exception as e:
    print(f"Error restoring model: {e}")