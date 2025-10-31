import pandas as pd

class MyClass:
    pass

instance = MyClass()
error = pd.errors.AbstractMethodError(instance, methodtype='classmethod')
print(str(error))