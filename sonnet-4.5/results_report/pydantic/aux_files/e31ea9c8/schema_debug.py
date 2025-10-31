from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated
import json

class Model(BaseModel):
    value: Annotated[str, transform(str.lower).str_upper()]

# Get the core schema
schema = Model.__pydantic_core_schema__
print("Core schema generated:")
print(json.dumps(schema, indent=2, default=str))

# Test the behavior
m = Model(value="ABC")
print(f"\nInput: 'ABC'")
print(f"After processing: '{m.value}'")