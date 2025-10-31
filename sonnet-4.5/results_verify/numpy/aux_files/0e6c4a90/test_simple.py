import numpy.typing as npt
import importlib

importlib.reload(npt)

delattr(npt, 'NBitBase')

obj = npt.NBitBase
print(f"obj = {obj}")