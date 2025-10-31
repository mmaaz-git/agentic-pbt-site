import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler import Options
import copy

def backup_Options():
    backup = {}
    for name, value in vars(Options).items():
        # we need a deep copy of _directive_defaults, because they can be changed
        if name == '_directive_defaults':
            value = copy.deepcopy(value)
        backup[name] = value
    return backup

def restore_Options_fixed(backup):
    no_value = object()
    for name, orig_value in backup.items():
        if getattr(Options, name, no_value) != orig_value:
            setattr(Options, name, orig_value)
    # strip Options from new keys that might have been added:
    # FIX: Convert keys() to a list to avoid RuntimeError
    for name in list(vars(Options).keys()):
        if name not in backup:
            delattr(Options, name)

print("Creating backup of Options...")
backup = backup_Options()
print(f"Backup created with {len(backup)} keys")

print("\nAdding new attribute to Options...")
Options.new_attribute = "test_value"
print(f"Options.new_attribute = {Options.new_attribute}")

print("\nAttempting to restore Options with fixed version...")
try:
    restore_Options_fixed(backup)
    print("restore_Options_fixed succeeded")
    print(f"Options has new_attribute: {hasattr(Options, 'new_attribute')}")
except RuntimeError as e:
    print(f"RuntimeError occurred: {e}")
except Exception as e:
    print(f"Unexpected exception: {type(e).__name__}: {e}")