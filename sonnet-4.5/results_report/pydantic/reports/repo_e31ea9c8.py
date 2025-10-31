from pydantic import BaseModel
from pydantic.experimental.pipeline import transform
from typing import Annotated

class Model(BaseModel):
    value: Annotated[str, transform(str.lower).str_upper()]

m = Model(value="ABC")
print(f"Input: 'ABC'")
print(f"Expected output: 'ABC' (first apply lower() to get 'abc', then apply upper() to get 'ABC')")
print(f"Actual output: '{m.value}'")