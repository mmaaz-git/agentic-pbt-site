from pandas.core.computation.scope import Scope

scope = Scope(level=0, global_dict={'old_key': 'old_value'})

print(f"Before: 'old_key' in scope.scope = {'old_key' in scope.scope}")
print(f"Before: 'new_key' in scope.scope = {'new_key' in scope.scope}")

scope.swapkey('old_key', 'new_key', 'new_value')

print(f"After: 'old_key' in scope.scope = {'old_key' in scope.scope}")
print(f"After: 'new_key' in scope.scope = {'new_key' in scope.scope}")
print(f"After: scope.scope['new_key'] = {scope.scope['new_key']}")

assert 'old_key' not in scope.scope, "old_key should have been removed but it still exists!"