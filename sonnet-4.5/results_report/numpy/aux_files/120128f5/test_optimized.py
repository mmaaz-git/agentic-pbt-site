import numpy.f2py.symbolic as symbolic

# Test with optimization flag (-O disables assertions)
print("Testing with -O flag (assertions disabled):")
result = symbolic.eliminate_quotes('"')
print(f"Result for single double quote: {result}")
print(f"The quote remains in output: {'"' in result[0]}")

result2 = symbolic.eliminate_quotes("'")
print(f"Result for single single quote: {result2}")
single_quote_check = "'" in result2[0]
print(f"The quote remains in output: {single_quote_check}")