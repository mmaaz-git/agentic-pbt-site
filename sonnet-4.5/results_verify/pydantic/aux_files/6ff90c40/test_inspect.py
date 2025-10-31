import pydantic
from pydantic import BaseModel, Field
import inspect

# Check the docstring for model_dump_json
print("=== model_dump_json docstring ===")
print(BaseModel.model_dump_json.__doc__)

# Let's also check populate_by_name behavior
print("\n=== Testing populate_by_name ===")

class ModelWithAliasPopulate(BaseModel):
    model_config = pydantic.ConfigDict(populate_by_name=True)
    field_one: str = Field(alias="fieldOne")

model = ModelWithAliasPopulate(fieldOne="test")
json_str = model.model_dump_json(by_alias=False, round_trip=True)
print(f"JSON with by_alias=False: {json_str}")

try:
    restored = ModelWithAliasPopulate.model_validate_json(json_str)
    print(f"Restored successfully with populate_by_name=True: {restored}")
except Exception as e:
    print(f"Error: {e}")