from pydantic.experimental.pipeline import _apply_constraint
from pydantic_core import core_schema as cs
import annotated_types

int_schema = cs.int_schema()

ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(5))
gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(5))

print(f"Ge(5) schema type: {ge_result['type']}")
print(f"Gt(5) schema type: {gt_result['type']}")

# Let's also check Le and Lt for completeness
le_result = _apply_constraint(int_schema.copy(), annotated_types.Le(5))
lt_result = _apply_constraint(int_schema.copy(), annotated_types.Lt(5))

print(f"Le(5) schema type: {le_result['type']}")
print(f"Lt(5) schema type: {lt_result['type']}")

# Let's see the actual schemas
print("\nGe(5) schema:", ge_result)
print("\nGt(5) schema:", gt_result)