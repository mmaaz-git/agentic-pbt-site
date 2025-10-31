import pandas.errors


class MyClass:
    pass


instance = MyClass()
error = pandas.errors.AbstractMethodError(instance, methodtype='classmethod')

print(str(error))