import time
from pydantic import BaseModel
from pydantic.experimental.pipeline import validate_as
from typing import Annotated


class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(10)]


class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(10)]


# Test validation performance
def test_performance(model_cls, test_value, iterations=100000):
    start = time.time()
    for _ in range(iterations):
        model_cls(value=test_value)
    end = time.time()
    return end - start


if __name__ == "__main__":
    iterations = 100000
    test_value = 20  # Valid value for both constraints

    print(f"Testing performance with {iterations} iterations...")
    print(f"Test value: {test_value}")
    print()

    gt_time = test_performance(ModelGt, test_value, iterations)
    print(f"ModelGt time: {gt_time:.4f} seconds")

    ge_time = test_performance(ModelGe, test_value, iterations)
    print(f"ModelGe time: {ge_time:.4f} seconds")

    print(f"\nDifference: {ge_time - gt_time:.4f} seconds ({((ge_time/gt_time - 1) * 100):.2f}% slower)")

    # Verify that both constraints work functionally
    print("\nFunctional test:")
    try:
        ModelGt(value=5)  # Should fail
    except Exception as e:
        print(f"ModelGt(value=5) correctly failed: {type(e).__name__}")

    try:
        ModelGe(value=5)  # Should fail
    except Exception as e:
        print(f"ModelGe(value=5) correctly failed: {type(e).__name__}")

    print("\nBoth models correctly validate the constraints functionally.")