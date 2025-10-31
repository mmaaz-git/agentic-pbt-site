import numpy.f2py.symbolic as sym
from numpy.f2py.symbolic import Language

# In C, ** is not a valid operator, so x**2 gets parsed as x * (*2)
# Let me verify this

print("Testing C parsing of 'x**2':")
e = sym.fromstring('x**2', language=Language.C)
print(f"Result: {e}")
print(f"tostring: {e.tostring()}")

# Let's manually check what C should do
# In C, there is no ** operator, so it would need to use pow()
print("\n--- What C should produce ---")
print("For x**2 in C, it should be: pow(x, 2)")

# Now let's check the tostring with different languages
print("\n--- tostring with different languages ---")
for lang in [Language.Fortran, Language.Python, Language.C]:
    print(f"{lang.name}: {e.tostring(language=lang)}")

# The parsing seems wrong for C - it interprets ** as * and *
# Let's see if there's documentation about this