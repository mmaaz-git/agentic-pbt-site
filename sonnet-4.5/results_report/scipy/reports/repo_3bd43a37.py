from pydantic import BaseModel, Field

class ModelWithAlias(BaseModel):
    field_one: str = Field(alias="fieldOne")

model = ModelWithAlias(fieldOne="test")

json_str = model.model_dump_json(by_alias=False, round_trip=True)
print(f"JSON: {json_str}")

try:
    restored = ModelWithAlias.model_validate_json(json_str)
    print(f"Successfully restored: {restored}")
except Exception as e:
    print(f"Error: {e}")