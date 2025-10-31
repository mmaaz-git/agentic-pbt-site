from pydantic import BaseModel
from typing import Annotated
from pydantic.experimental.pipeline import validate_as

pipeline = validate_as(str).str_strip()

class TestModel(BaseModel):
    field: Annotated[str, pipeline]

test_input = '0\x1f'
print(f"Input: {test_input!r}")
print(f"Python's str.strip(): {test_input.strip()!r}")

model = TestModel(field=test_input)
print(f"Pipeline str_strip(): {model.field!r}")
print(f"Match: {model.field == test_input.strip()}")

# Also test with just the \x1f character
test_input2 = '\x1f'
print(f"\nInput: {test_input2!r}")
print(f"Python's str.strip(): {test_input2.strip()!r}")

model2 = TestModel(field=test_input2)
print(f"Pipeline str_strip(): {model2.field!r}")
print(f"Match: {model2.field == test_input2.strip()}")