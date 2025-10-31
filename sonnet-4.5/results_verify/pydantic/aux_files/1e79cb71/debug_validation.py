import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/pydantic_env/lib/python3.13/site-packages')

from pydantic import BaseModel, ValidationError
from pydantic.experimental.pipeline import transform
import operator

# First, let's test if the constraint function is actually being called
print("Testing not_in constraint validation:")

class Model(BaseModel):
    value: int = transform(lambda v: v).not_in([5, 10, 15])

# Try values that should pass
for val in [1, 7, 20]:
    try:
        m = Model(value=val)
        print(f"✓ {val} passed (expected)")
    except ValidationError as e:
        print(f"✗ {val} failed (unexpected): {e}")

# Try values that should fail
for val in [5, 10, 15]:
    try:
        m = Model(value=val)
        print(f"✗ {val} passed (unexpected - BUG!)")
    except ValidationError as e:
        print(f"✓ {val} failed (expected): {e}")

# Also let's test the 'in' constraint for comparison
print("\n\nTesting 'in' constraint for comparison:")

class ModelIn(BaseModel):
    value: int = transform(lambda v: v).in_([5, 10, 15])

# Try values that should fail (not in list)
for val in [1, 7, 20]:
    try:
        m = ModelIn(value=val)
        print(f"✗ {val} passed (unexpected)")
    except ValidationError as e:
        print(f"✓ {val} failed (expected)")

# Try values that should pass (in list)
for val in [5, 10, 15]:
    try:
        m = ModelIn(value=val)
        print(f"✓ {val} passed (expected)")
    except ValidationError as e:
        print(f"✗ {val} failed (unexpected): {e}")