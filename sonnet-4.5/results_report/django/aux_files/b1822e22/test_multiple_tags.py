from django.core.checks.registry import CheckRegistry
from django.core.checks import Info

registry = CheckRegistry()

def check(**kwargs):
    return []

# Test registering with multiple tags at once
registry.register(check, 'tag1', 'tag2')
print('Tags when registered with multiple tags at once:', check.tags)

available_tags = registry.tags_available()
print('Available tags:', available_tags)

# Run checks with each tag
checks_tag1 = registry.run_checks(tags=["tag1"])
checks_tag2 = registry.run_checks(tags=["tag2"])
print(f"Checks with tag1: {len(checks_tag1)}")
print(f"Checks with tag2: {len(checks_tag2)}")