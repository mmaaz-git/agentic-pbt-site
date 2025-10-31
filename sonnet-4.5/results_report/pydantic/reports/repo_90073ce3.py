import annotated_types
from pydantic_core import core_schema as cs
from pydantic.experimental.pipeline import _apply_constraint

# Create a basic integer schema
int_schema = cs.int_schema()

# Apply Gt constraint (correctly implemented)
gt_result = _apply_constraint(int_schema.copy(), annotated_types.Gt(10))
print("Gt constraint result:")
print(f"  Schema type: {gt_result['type']}")
print(f"  Full schema: {gt_result}")
print()

# Apply Ge constraint (buggy - applies twice)
ge_result = _apply_constraint(int_schema.copy(), annotated_types.Ge(10))
print("Ge constraint result:")
print(f"  Schema type: {ge_result['type']}")
if ge_result['type'] == 'function-after':
    print(f"  Inner schema: {ge_result['schema']}")
print(f"  Full schema: {ge_result}")
print()

# Apply Lt constraint (buggy - applies twice)
lt_result = _apply_constraint(int_schema.copy(), annotated_types.Lt(100))
print("Lt constraint result:")
print(f"  Schema type: {lt_result['type']}")
if lt_result['type'] == 'function-after':
    print(f"  Inner schema: {lt_result['schema']}")
print(f"  Full schema: {lt_result}")
print()

# Apply Le constraint (buggy - applies twice)
le_result = _apply_constraint(int_schema.copy(), annotated_types.Le(100))
print("Le constraint result:")
print(f"  Schema type: {le_result['type']}")
if le_result['type'] == 'function-after':
    print(f"  Inner schema: {le_result['schema']}")
print(f"  Full schema: {le_result}")
print()

# Apply MultipleOf constraint (buggy - applies twice)
multiple_result = _apply_constraint(int_schema.copy(), annotated_types.MultipleOf(5))
print("MultipleOf constraint result:")
print(f"  Schema type: {multiple_result['type']}")
if multiple_result['type'] == 'function-after':
    print(f"  Inner schema: {multiple_result['schema']}")
print(f"  Full schema: {multiple_result}")