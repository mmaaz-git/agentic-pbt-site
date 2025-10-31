import copy
from Cython.Compiler import Options

def backup_Options():
    backup = {}
    for name, value in vars(Options).items():
        # we need a deep copy of _directive_defaults, because they can be changed
        if name == '_directive_defaults':
            value = copy.deepcopy(value)
        backup[name] = value
    return backup

def restore_Options_fixed(backup):
    """Fixed version of restore_Options"""
    no_value = object()
    for name, orig_value in backup.items():
        if getattr(Options, name, no_value) != orig_value:
            setattr(Options, name, orig_value)
    # strip Options from new keys that might have been added:
    # FIX: Convert keys() to a list to avoid RuntimeError
    for name in list(vars(Options).keys()):
        if name not in backup:
            delattr(Options, name)

print("Testing fixed version...")
backup = backup_Options()

# Add a new attribute
Options.TestAttr = "test_value"
print(f"Added new attribute Options.TestAttr = {Options.TestAttr}")

# Try to restore with fixed version
try:
    restore_Options_fixed(backup)
    print("restore_Options_fixed succeeded")

    # Check if the attribute was removed
    if hasattr(Options, 'TestAttr'):
        print("ERROR: Attribute TestAttr still exists after restore!")
    else:
        print("SUCCESS: Attribute TestAttr was removed after restore")
except RuntimeError as e:
    print(f"RuntimeError still occurred: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")