import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from pydantic import BaseModel
from pydantic.experimental.pipeline import transform

# Create a counter to track validation calls
validation_count = {'ge': 0, 'gt': 0}

# Monkey patch to track validation calls
original_ge = lambda v, threshold: v >= threshold
original_gt = lambda v, threshold: v > threshold

def tracked_ge(v):
    validation_count['ge'] += 1
    return v >= 5

def tracked_gt(v):
    validation_count['gt'] += 1
    return v > 4

# Test with transformation that tracks calls
class ModelGe(BaseModel):
    value: int = transform(lambda v: v).constrain(lambda x: (tracked_ge(x), x)[1])

class ModelGt(BaseModel):
    value: int = transform(lambda v: v).constrain(lambda x: (tracked_gt(x), x)[1])

# Actually, let me trace this differently - let's see how many times validators are called
import pydantic.experimental.pipeline as pipeline

original_check_func = pipeline._check_func
call_counts = {}

def tracked_check_func(func, message, s):
    if message not in call_counts:
        call_counts[message] = 0
    call_counts[message] += 1
    print(f"_check_func called with message: {message}")
    return original_check_func(func, message, s)

pipeline._check_func = tracked_check_func

# Now test the models
from pydantic import BaseModel
from pydantic.experimental.pipeline import transform

print("Testing Ge constraint:")
class ModelGe(BaseModel):
    value: int = transform(lambda v: v).ge(5)

print("\nTesting Gt constraint:")
class ModelGt(BaseModel):
    value: int = transform(lambda v: v).gt(4)

print("\nTesting Lt constraint:")
class ModelLt(BaseModel):
    value: int = transform(lambda v: v).lt(100)

print("\nTesting Le constraint:")
class ModelLe(BaseModel):
    value: int = transform(lambda v: v).le(100)

print("\nTesting MultipleOf constraint:")
class ModelMultiple(BaseModel):
    value: int = transform(lambda v: v).multiple_of(5)

print("\nCall counts summary:")
for msg, count in call_counts.items():
    print(f"  {msg}: called {count} time(s)")