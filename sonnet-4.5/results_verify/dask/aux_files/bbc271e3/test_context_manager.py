from dask.callbacks import Callback, add_callbacks

# Test that the add_callbacks context manager handles this properly
print("Testing add_callbacks context manager behavior...")

cb1 = Callback()
cb2 = Callback()

print(f"cb1._callback: {cb1._callback}")
print(f"cb2._callback: {cb2._callback}")

print("\nUsing add_callbacks with both callbacks:")
print(f"Initial Callback.active: {Callback.active}")

with add_callbacks(cb1, cb2):
    print(f"Inside context: Callback.active = {Callback.active}")
    print(f"Length of active: {len(Callback.active)}")

print(f"After context: Callback.active = {Callback.active}")
print("No errors occurred - add_callbacks uses discard() which handles duplicates gracefully")