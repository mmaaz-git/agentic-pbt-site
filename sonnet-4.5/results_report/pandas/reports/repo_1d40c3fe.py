import pandas.core.computation.parsing as parsing

# Test case with null byte
name = '\x00'
result = parsing.clean_column_name(name)