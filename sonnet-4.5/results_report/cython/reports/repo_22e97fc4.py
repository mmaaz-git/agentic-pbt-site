import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options

# Create a backup of the current Options state
backup = backup_Options()

# Add a new attribute to Options
Options.new_attribute = "test_value"

# Try to restore the Options to its original state
# This should remove the new_attribute we just added
restore_Options(backup)