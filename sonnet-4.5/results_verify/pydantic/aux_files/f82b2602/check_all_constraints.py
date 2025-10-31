from pydantic.experimental.pipeline import _apply_constraint
from pydantic_core import core_schema as cs
import annotated_types

# Test with int schema
int_schema = cs.int_schema()

# Test all four comparison constraints
constraints = [
    ('Gt', annotated_types.Gt(5)),
    ('Ge', annotated_types.Ge(5)),
    ('Lt', annotated_types.Lt(5)),
    ('Le', annotated_types.Le(5))
]

print("Results for int schema:")
for name, constraint in constraints:
    result = _apply_constraint(int_schema.copy(), constraint)
    print(f"{name}(5) schema type: {result['type']}")
    if result['type'] == 'function-after':
        print(f"  Inner schema: {result['schema']}")
    else:
        print(f"  Direct schema: {result}")

# Test with a schema that doesn't support these constraints directly
str_schema = cs.str_schema()
print("\nResults for str schema (should all be function-after):")
for name, constraint in constraints:
    result = _apply_constraint(str_schema.copy(), constraint)
    print(f"{name}(5) on str schema type: {result['type']}")