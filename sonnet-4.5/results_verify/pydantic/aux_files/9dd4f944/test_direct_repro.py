#!/usr/bin/env python3
"""Direct reproduction from bug report"""

from pydantic.v1 import BaseModel, Field

class MultipleOf(BaseModel):
    value: int = Field(multiple_of=5)

value = 10**16 + 1

model = MultipleOf(value=value)
print(f"Accepted: {model.value}")
print(f"Value % 5 = {model.value % 5}")

assert model.value % 5 == 0