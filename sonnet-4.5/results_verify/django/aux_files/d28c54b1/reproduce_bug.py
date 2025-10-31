import sys
import os

# Add Django environment to path
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'django.conf.global_settings')

import django
django.setup()

from django.db.backends.utils import truncate_name, split_identifier

identifier = '00'
length = 1

result = truncate_name(identifier, length=length)
namespace, name = split_identifier(result)

print(f"truncate_name('{identifier}', length={length})")
print(f"Expected: name with length <= {length}")
print(f"Actual: '{result}' (name part: '{name}', length = {len(name)})")
print(f"Bug: The function returns a name of length {len(name)}, which exceeds the requested length of {length}")

# Test a few more cases
print("\n--- Additional test cases ---")
for test_id, test_len in [('a', 1), ('ab', 2), ('abc', 3), ('abcd', 4), ('abcde', 5)]:
    result = truncate_name(test_id, length=test_len)
    _, name = split_identifier(result)
    print(f"truncate_name('{test_id}', length={test_len}) -> '{result}' (name length: {len(name)})")
    if len(name) > test_len:
        print(f"  ⚠️ VIOLATION: {len(name)} > {test_len}")
