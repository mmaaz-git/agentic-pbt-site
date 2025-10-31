import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

import Cython.Compiler.Tests.Utils as Utils
import Cython.Compiler.Options as Options

# Save the original state
original_backup = Utils.backup_Options()

# Add a new attribute to Options
Options.new_test_attr = "test_value"

# Create a backup with the new attribute included
modified_backup = Utils.backup_Options()

# Try to restore the original state (this should crash)
print("Attempting to restore original Options state...")
Utils.restore_Options(original_backup)
print("Restore successful!")