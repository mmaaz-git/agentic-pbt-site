import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options

print("Creating backup of Options...")
backup = backup_Options()
print(f"Backup created with {len(backup)} keys")

print("\nAdding new attribute to Options...")
Options.new_attribute = "test_value"
print(f"Options.new_attribute = {Options.new_attribute}")

print("\nAttempting to restore Options...")
try:
    restore_Options(backup)
    print("restore_Options succeeded")
    print(f"Options has new_attribute: {hasattr(Options, 'new_attribute')}")
except RuntimeError as e:
    print(f"RuntimeError occurred: {e}")
except Exception as e:
    print(f"Unexpected exception: {type(e).__name__}: {e}")