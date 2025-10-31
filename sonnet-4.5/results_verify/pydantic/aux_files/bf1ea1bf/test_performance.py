#!/usr/bin/env python3
"""Test performance impact of double validation"""

from typing import Annotated
from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import validate_as
import time

class ModelGt(BaseModel):
    value: Annotated[int, validate_as(int).gt(5)]

class ModelGe(BaseModel):
    value: Annotated[int, validate_as(int).ge(5)]

# Test with many validation attempts to measure performance difference
iterations = 100000

print("Performance test with valid values...")
print("-" * 40)

# Test with valid value
start = time.perf_counter()
for i in range(iterations):
    ModelGt(value=10)
gt_time = time.perf_counter() - start

start = time.perf_counter()
for i in range(iterations):
    ModelGe(value=10)
ge_time = time.perf_counter() - start

print(f"Gt (single validation): {gt_time:.4f} seconds")
print(f"Ge (double validation): {ge_time:.4f} seconds")
print(f"Ge is {ge_time/gt_time:.2f}x slower")

print("\nPerformance test with invalid values...")
print("-" * 40)

# Test with invalid value (measure exception handling cost)
start = time.perf_counter()
for i in range(iterations):
    try:
        ModelGt(value=3)
    except ValidationError:
        pass
gt_time = time.perf_counter() - start

start = time.perf_counter()
for i in range(iterations):
    try:
        ModelGe(value=3)
    except ValidationError:
        pass
ge_time = time.perf_counter() - start

print(f"Gt (single validation): {gt_time:.4f} seconds")
print(f"Ge (double validation): {ge_time:.4f} seconds")
print(f"Ge is {ge_time/gt_time:.2f}x slower")