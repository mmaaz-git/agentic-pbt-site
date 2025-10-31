from pydantic.experimental.pipeline import _apply_constraint
from pydantic_core import core_schema as cs
import annotated_types

# Create an integer schema
int_schema = cs.int_schema()

# Apply Ge(5) constraint
ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(5))
print(f"Ge(5) result:")
print(f"  Schema type: {ge_result['type']}")
print(f"  Full schema: {ge_result}")
print()

# Apply Gt(5) constraint
gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(5))
print(f"Gt(5) result:")
print(f"  Schema type: {gt_result['type']}")
print(f"  Full schema: {gt_result}")
print()

# Apply Lt(5) constraint
lt_result = _apply_constraint(int_schema.copy(), annotated_types.Lt(5))
print(f"Lt(5) result:")
print(f"  Schema type: {lt_result['type']}")
print(f"  Full schema: {lt_result}")
print()

# Apply Le(5) constraint
le_result = _apply_constraint(int_schema.copy(), annotated_types.Le(5))
print(f"Le(5) result:")
print(f"  Schema type: {le_result['type']}")
print(f"  Full schema: {le_result}")
print()

# Show the inconsistency
print("Comparison:")
print(f"  Gt produces simple schema: {gt_result['type'] == 'int'}")
print(f"  Ge produces wrapped schema: {ge_result['type'] == 'function-after'}")
print(f"  Lt produces wrapped schema: {lt_result['type'] == 'function-after'}")
print(f"  Le produces wrapped schema: {le_result['type'] == 'function-after'}")