from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options

print("Testing simple reproduction case...")
backup = backup_Options()
print("Backup created")

# Add a new attribute
Options.A = "test_value"
print(f"Added new attribute Options.A = {Options.A}")

# Try to restore
try:
    restore_Options(backup)
    print("restore_Options succeeded")

    # Check if the attribute was removed
    if hasattr(Options, 'A'):
        print("ERROR: Attribute A still exists after restore!")
    else:
        print("SUCCESS: Attribute A was removed after restore")
except RuntimeError as e:
    print(f"RuntimeError occurred: {e}")
    print("BUG CONFIRMED: Dictionary changed size during iteration")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")