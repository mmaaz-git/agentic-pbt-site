from django.core.checks.registry import CheckRegistry
from django.core.checks import Info

registry = CheckRegistry()

def my_check(app_configs=None, **kwargs):
    return [Info("My check")]

# First registration with tag1
registry.register(my_check, "tag1")
print(f"After first registration - tags: {my_check.tags}")

# Second registration with tag2
registry.register(my_check, "tag2")
print(f"After second registration - tags: {my_check.tags}")

# Check available tags
available_tags = registry.tags_available()
print(f"Available tags: {available_tags}")

# Try to run checks with each tag
checks_tag1 = registry.run_checks(tags=["tag1"])
checks_tag2 = registry.run_checks(tags=["tag2"])

print(f"Checks with tag1: {len(checks_tag1)}")
print(f"Checks with tag2: {len(checks_tag2)}")