from Cython.Compiler.Tests.Utils import backup_Options, restore_Options
from Cython.Compiler import Options

backup = backup_Options()
Options.A = "test_value"
restore_Options(backup)