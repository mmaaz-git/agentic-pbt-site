from django.template import Variable

# Create a Variable with a number followed by trailing period
var = Variable("10.")

# Display the internal state - both literal and lookups are set
print(f"literal: {var.literal}")
print(f"lookups: {var.lookups}")

# This shows the inconsistent state - both are set when they should be mutually exclusive
assert var.literal == 10.0, f"Expected literal to be 10.0, got {var.literal}"
assert var.lookups == ('10', ''), f"Expected lookups to be ('10', ''), got {var.lookups}"

# Try to resolve the variable - this raises an exception instead of returning the literal
try:
    result = var.resolve({})
    print(f"Result: {result}")
except Exception as e:
    print(f"{type(e).__name__}: {e}")