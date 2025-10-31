import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

import django.template

v = django.template.Variable('2.')

print(f'literal: {v.literal}')
print(f'lookups: {v.lookups}')

c = django.template.Context({})
try:
    result = v.resolve(c)
    print(f'Result: {result}')
except django.template.VariableDoesNotExist as e:
    print(f'ERROR: {e}')