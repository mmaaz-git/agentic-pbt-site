from pydantic import BaseModel
from typing import Annotated
from pydantic.experimental.pipeline import transform

# First example: String input with integer constraint
int_pipeline = transform(lambda x: x).ge(0)
str_pipeline = transform(lambda x: x).str_lower()
combined_pipeline = int_pipeline.otherwise(str_pipeline)

class Model(BaseModel):
    x: Annotated[int | str, combined_pipeline]

print("Testing with string 'HELLO':")
try:
    result = Model(x='HELLO')
    print(f"Success: {result.x}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting with negative integer -1:")
try:
    result = Model(x=-1)
    print(f"Success: {result.x}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting with positive integer 5:")
try:
    result = Model(x=5)
    print(f"Success: {result.x}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")

print("\nTesting with empty string '':")
try:
    result = Model(x='')
    print(f"Success: {result.x}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")