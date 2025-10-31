import pandas.api.typing
import inspect

# Check NAType __new__ method
print("NAType.__new__ source:")
try:
    print(inspect.getsource(pandas.api.typing.NAType.__new__))
except:
    print("Can't get source - likely implemented in C")

# Let's see if NAType has a custom __new__ that returns singleton
na1 = pandas.api.typing.NAType()
na2 = pandas.api.typing.NAType()
print(f"\nNAType() returns same object: {na1 is na2}")

# Check NaTType __new__ method
print("\nNaTType.__new__ source:")
try:
    print(inspect.getsource(pandas.api.typing.NaTType.__new__))
except:
    print("Can't get source - likely implemented in C")

nat1 = pandas.api.typing.NaTType()
nat2 = pandas.api.typing.NaTType()
print(f"\nNaTType() returns same object: {nat1 is nat2}")