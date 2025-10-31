from django.template import Variable, Context

# Test the bug with "2."
v = Variable("2.")

print(f"literal = {v.literal}")
print(f"lookups = {v.lookups}")

# Create a context that would be matched by the lookups
ctx = Context({"2": {"": "unexpected_value"}})
result = v.resolve(ctx)
print(f"Result when resolving '2.': {result}")

# For comparison, test normal float and integer
v_normal_float = Variable("2.0")
print(f"\n'2.0': literal={v_normal_float.literal}, lookups={v_normal_float.lookups}")

v_integer = Variable("2")
print(f"'2': literal={v_integer.literal}, lookups={v_integer.lookups}")