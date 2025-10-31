import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/cython_env/lib/python3.13/site-packages')

from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options

print("Starting test...")
backup = backup_Options()
print("Backup created")
Options.new_test_attr = "test_value"
print("Added new attribute")
try:
    restore_Options(backup)
    print("restore_Options completed successfully")
except RuntimeError as e:
    print(f"RuntimeError occurred: {e}")
except Exception as e:
    print(f"Other exception occurred: {e}")