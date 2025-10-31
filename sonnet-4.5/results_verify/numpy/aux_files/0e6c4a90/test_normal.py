import numpy.typing as npt
import importlib

importlib.reload(npt)

# Check if NBitBase exists normally
print("NBitBase in dir(npt):", "NBitBase" in dir(npt))
print("NBitBase in npt.__dict__:", "NBitBase" in npt.__dict__)
print("NBitBase in globals() within module:", "NBitBase" in vars(npt))

# Try to access it normally
try:
    obj = npt.NBitBase
    print(f"obj = {obj}")
except Exception as e:
    print(f"Error accessing NBitBase: {e}")