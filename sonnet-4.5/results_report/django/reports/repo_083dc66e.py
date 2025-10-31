import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from django.core.checks.registry import CheckRegistry

# Create a registry instance
registry = CheckRegistry()

# Define a buggy check function that returns a string instead of a list
def buggy_check(app_configs=None, **kwargs):
    return "error message"

# Register the buggy check
registry.register(buggy_check)

# Run the checks
errors = registry.run_checks()

# Display the errors
print(f"Errors list: {errors}")
print(f"Errors type: {type(errors)}")
print(f"Number of items in errors: {len(errors)}")
print(f"First item: {repr(errors[0]) if errors else 'No errors'}")
print(f"All items (repr): {[repr(e) for e in errors]}")