from pydantic import BaseModel
from typing import Annotated
from pydantic.experimental.pipeline import transform

# First test case - string input
print("Test 1: String input 'HELLO'")
try:
    int_pipeline = transform(lambda x: x).ge(0)
    str_pipeline = transform(lambda x: x).str_lower()
    combined_pipeline = int_pipeline.otherwise(str_pipeline)

    class Model(BaseModel):
        x: Annotated[int | str, combined_pipeline]

    result = Model(x='HELLO')
    print(f"Result: {result.x}")
except TypeError as e:
    print(f"TypeError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Second test case - negative integer
print("Test 2: Negative integer -1")
try:
    int_pipeline = transform(lambda x: x).ge(0)
    str_pipeline = transform(lambda x: x).str_lower()
    combined_pipeline = int_pipeline.otherwise(str_pipeline)

    class Model2(BaseModel):
        x: Annotated[int | str, combined_pipeline]

    result2 = Model2(x=-1)
    print(f"Result: {result2.x}")
except TypeError as e:
    print(f"TypeError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Third test case - positive integer (should work)
print("Test 3: Positive integer 5")
try:
    int_pipeline = transform(lambda x: x).ge(0)
    str_pipeline = transform(lambda x: x).str_lower()
    combined_pipeline = int_pipeline.otherwise(str_pipeline)

    class Model3(BaseModel):
        x: Annotated[int | str, combined_pipeline]

    result3 = Model3(x=5)
    print(f"Result: {result3.x}")
except TypeError as e:
    print(f"TypeError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")

print("\n" + "="*50 + "\n")

# Fourth test case - lowercase string (should work)
print("Test 4: Lowercase string 'hello'")
try:
    int_pipeline = transform(lambda x: x).ge(0)
    str_pipeline = transform(lambda x: x).str_lower()
    combined_pipeline = int_pipeline.otherwise(str_pipeline)

    class Model4(BaseModel):
        x: Annotated[int | str, combined_pipeline]

    result4 = Model4(x='hello')
    print(f"Result: {result4.x}")
except TypeError as e:
    print(f"TypeError: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")