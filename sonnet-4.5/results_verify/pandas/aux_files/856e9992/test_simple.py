from Cython.Utility import pylong_join, _pylong_join

result1 = pylong_join(0)
result2 = _pylong_join(0)

print(f"pylong_join(0):  {result1!r}")
print(f"_pylong_join(0): {result2!r}")

assert result1 == result2