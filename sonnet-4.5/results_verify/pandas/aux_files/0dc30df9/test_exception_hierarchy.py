import tokenize

# Check if TokenError is a subclass of SyntaxError
print(f"TokenError is subclass of SyntaxError: {issubclass(tokenize.TokenError, SyntaxError)}")
print(f"TokenError MRO: {tokenize.TokenError.__mro__}")