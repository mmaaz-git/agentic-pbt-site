from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import validate_as
from typing import Annotated


class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(5.5)]


try:
    m = ModelGt(value=5)
    print(f"BUG: value=5 passed gt(5.5) validation! Result: {m.value}")
except ValidationError as e:
    print(f"Expected behavior: {e}")


class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(5.5)]


try:
    m = ModelGe(value=5)
    print(f"BUG: value=5 passed ge(5.5) validation!")
except ValidationError as e:
    print(f"Correct: ge(5.5) properly rejects 5")