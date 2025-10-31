from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options

original_backup = backup_Options()
setattr(Options, 'new_test_key', 'new_test_value')

restore_Options(original_backup)