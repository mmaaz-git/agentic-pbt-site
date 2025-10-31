import numpy.typing as npt

# Try the bug reproduction from the report
print("Testing the NBitBase __getattr__ bug...")

# Delete NBitBase from module dict to force __getattr__ to be called
del npt.__dict__['NBitBase']

try:
    # This should call __getattr__ which should return NBitBase
    obj = npt.NBitBase
    print(f"Successfully got NBitBase: {obj}")
except NameError as e:
    print(f"NameError occurred: {e}")
except AttributeError as e:
    print(f"AttributeError occurred: {e}")
except Exception as e:
    print(f"Unexpected error: {type(e).__name__}: {e}")