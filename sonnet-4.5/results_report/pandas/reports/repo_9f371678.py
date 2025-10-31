import pandas.core.computation.parsing as parsing

name = '\x00'
result = parsing.clean_column_name(name)
print(f"Result: {result}")