import django.template

# Create a Variable with a trailing dot
v = django.template.Variable('2.')

# Inspect the state of the Variable
print(f'Variable created with input: "2."')
print(f'literal: {v.literal}')
print(f'lookups: {v.lookups}')
print('')

# Both literal and lookups are set - this violates the invariant
if v.literal is not None and v.lookups is not None:
    print('BUG: Both literal and lookups are set!')
    print('This violates the class invariant that a Variable should be either a literal OR a lookup.')
print('')

# Try to resolve the Variable
c = django.template.Context({})
print('Attempting to resolve the Variable...')
try:
    result = v.resolve(c)
    print(f'Result: {result}')
except django.template.VariableDoesNotExist as e:
    print(f'ERROR - VariableDoesNotExist: {e}')