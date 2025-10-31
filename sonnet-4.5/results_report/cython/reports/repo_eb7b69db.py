from Cython.Compiler import Options

# This demonstrates the Unicode digit crash in parse_variable_value
result = Options.parse_variable_value('²')
print(f"Result: {result}")