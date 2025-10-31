import inspect
from scipy.odr._odrpack import ODR

# Get the source code for set_job and set_iprint methods
print("=== set_job source ===")
print(inspect.getsource(ODR.set_job))

print("\n=== set_iprint source ===")
print(inspect.getsource(ODR.set_iprint))