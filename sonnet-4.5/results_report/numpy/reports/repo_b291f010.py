import numpy.typing as npt

# Delete NBitBase from the module's __dict__ to force __getattr__ to be called
del npt.__dict__['NBitBase']

# Try to access NBitBase, which should trigger __getattr__
try:
    obj = npt.NBitBase
    print(f"Successfully retrieved: {obj}")
except NameError as e:
    print(f"NameError: {e}")