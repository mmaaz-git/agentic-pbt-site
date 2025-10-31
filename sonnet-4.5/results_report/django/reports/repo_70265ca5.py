from django.core.checks.registry import CheckRegistry


# Create a CheckRegistry instance
registry = CheckRegistry()


# Define a bad check function that returns a string instead of a list
def bad_check(app_configs, **kwargs):
    return "error message"


# Register the bad check
registry.register(bad_check, "test")

# Run the checks
result = registry.run_checks()

# Print the results
print(f"Result: {result}")
print(f"Type of result: {type(result)}")
print(f"Length of result: {len(result)}")
print(f"Expected: A list of CheckMessage objects")
print(f"Actual: A list of individual characters from the string")