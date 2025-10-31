import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import Cython.Compiler.Tests.Utils as Utils
import Cython.Compiler.Options as Options

print("Starting reproduction...")

original_backup = Utils.backup_Options()
print("Original backup created")

Options.new_test_attr = "test_value"
print("Added new attribute: new_test_attr = 'test_value'")

modified_backup = Utils.backup_Options()
print("Modified backup created")

try:
    Utils.restore_Options(original_backup)
    print("Successfully restored original options")
except RuntimeError as e:
    print(f"RuntimeError occurred: {e}")
except Exception as e:
    print(f"Other exception occurred: {type(e).__name__}: {e}")