import numpy.typing as npt
import importlib

# Reload to ensure a clean state
importlib.reload(npt)

# Delete NBitBase from the module namespace
delattr(npt, 'NBitBase')

# Try to access NBitBase through __getattr__
# This should trigger the deprecation warning and return NBitBase,
# but instead it will crash with NameError
obj = npt.NBitBase