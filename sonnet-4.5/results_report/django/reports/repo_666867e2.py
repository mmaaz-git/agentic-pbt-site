from django.template import Variable

# Test case that should raise an error but doesn't
var = Variable("2.")

print(f"Variable created successfully for '2.'")
print(f"literal: {var.literal}")
print(f"lookups: {var.lookups}")
print()

# Try with more trailing dot cases
test_cases = ["1.", "10.", "999.", "1234."]
for test_str in test_cases:
    var = Variable(test_str)
    print(f"Variable('{test_str}'): literal={var.literal}, lookups={var.lookups}")