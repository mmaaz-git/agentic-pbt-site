from dask.callbacks import Callback

print("Creating two Callback instances...")
cb1 = Callback()
cb2 = Callback()

print(f"cb1._callback: {cb1._callback}")
print(f"cb2._callback: {cb2._callback}")
print(f"Are they the same? {cb1._callback == cb2._callback}")

print("\nRegistering both callbacks...")
cb1.register()
print(f"Callback.active after cb1.register(): {Callback.active}")
cb2.register()
print(f"Callback.active after cb2.register(): {Callback.active}")

print("\nUnregistering cb1...")
cb1.unregister()
print(f"Callback.active after cb1.unregister(): {Callback.active}")

print("\nUnregistering cb2...")
try:
    cb2.unregister()
    print("cb2.unregister() succeeded")
except KeyError as e:
    print(f"KeyError raised: {e}")
    print("This is the bug - cb2.unregister() fails because the shared callback was already removed")