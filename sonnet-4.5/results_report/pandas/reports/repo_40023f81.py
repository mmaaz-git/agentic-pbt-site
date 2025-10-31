from numpy.f2py.symbolic import as_string, normalize

a = as_string('""', 1)
b = as_string('""', 1)
c = as_string("''", 1)

result1 = (a // b) // c
result2 = a // (b // c)

print(f"(a // b) // c = {repr(normalize(result1))}")
print(f"a // (b // c) = {repr(normalize(result2))}")
print(f"Equal: {normalize(result1) == normalize(result2)}")