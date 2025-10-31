from dask.callbacks import Callback

# Create two identical Callback instances
cb1 = Callback()
cb2 = Callback()

# Print their _callback tuples to show they are identical
print(f"cb1._callback: {cb1._callback}")
print(f"cb2._callback: {cb2._callback}")
print(f"Are they equal? {cb1._callback == cb2._callback}")
print()

# Register both callbacks
print("Registering cb1...")
cb1.register()
print(f"Callback.active after cb1.register(): {Callback.active}")

print("\nRegistering cb2...")
cb2.register()
print(f"Callback.active after cb2.register(): {Callback.active}")
print(f"Number of entries in active set: {len(Callback.active)}")

# Unregister the first callback
print("\nUnregistering cb1...")
cb1.unregister()
print(f"Callback.active after cb1.unregister(): {Callback.active}")

# Try to unregister the second callback - this will raise KeyError
print("\nUnregistering cb2...")
try:
    cb2.unregister()
    print("cb2 unregistered successfully")
except KeyError as e:
    print(f"KeyError raised: {e}")