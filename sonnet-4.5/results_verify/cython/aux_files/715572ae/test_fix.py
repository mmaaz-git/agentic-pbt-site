import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import copy

# Copy the original function and apply the fix
def backup_Options():
    import Cython.Compiler.Options as Options
    backup = {}
    for name, value in vars(Options).items():
        # we need a deep copy of _directive_defaults, because they can be changed
        if name == '_directive_defaults':
            value = copy.deepcopy(value)
        backup[name] = value
    return backup


def restore_Options_fixed(backup):
    import Cython.Compiler.Options as Options
    no_value = object()
    for name, orig_value in backup.items():
        if getattr(Options, name, no_value) != orig_value:
            setattr(Options, name, orig_value)
    # strip Options from new keys that might have been added:
    # FIX: Convert to list to avoid modifying dict during iteration
    for name in list(vars(Options).keys()):
        if name not in backup:
            delattr(Options, name)

import Cython.Compiler.Options as Options

print("Testing with the fix...")

original_backup = backup_Options()
print("Original backup created")

Options.new_test_attr = "test_value"
print("Added new attribute: new_test_attr = 'test_value'")

modified_backup = backup_Options()
print("Modified backup created")

try:
    restore_Options_fixed(original_backup)
    print("Successfully restored original options with the fix!")

    # Verify the new attribute was removed
    if not hasattr(Options, 'new_test_attr'):
        print("✓ New attribute correctly removed")
    else:
        print("✗ New attribute still exists")

except Exception as e:
    print(f"Error occurred: {type(e).__name__}: {e}")