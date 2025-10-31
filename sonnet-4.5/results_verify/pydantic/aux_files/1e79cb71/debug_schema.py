import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
import json

# Create models with constraints
class ModelNotIn(BaseModel):
    value: int = transform(lambda v: v).not_in([5, 10, 15])

class ModelIn(BaseModel):
    value: int = transform(lambda v: v).in_([5, 10, 15])

# Check the schema
print("ModelNotIn schema:")
print(json.dumps(ModelNotIn.model_json_schema(), indent=2))

print("\n\nModelIn schema:")
print(json.dumps(ModelIn.model_json_schema(), indent=2))

# Check the field info
print("\n\nModelNotIn field info:")
print(ModelNotIn.model_fields['value'])

print("\n\nModelIn field info:")
print(ModelIn.model_fields['value'])