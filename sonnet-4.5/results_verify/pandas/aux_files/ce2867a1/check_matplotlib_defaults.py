import matplotlib.units as munits
import datetime as pydt
import numpy as np

# Clean start
munits.registry.clear()
print("=== Starting with empty registry ===")
print(f"Initial registry: {list(munits.registry.keys())}")

# Check if matplotlib has default converters for these types
print("\n=== Checking matplotlib's default behavior ===")

# Try to access datetime converter - matplotlib might lazily register it
import matplotlib.pyplot as plt
fig, ax = plt.subplots()

# Plot something with datetime to trigger matplotlib's lazy registration
dates = [pydt.datetime(2020, 1, 1), pydt.datetime(2020, 1, 2)]
values = [1, 2]

# This should trigger matplotlib to register its own converters
ax.plot(dates, values)
print(f"After plotting datetime: {list(munits.registry.keys())}")

# Clear and try again
plt.close('all')
munits.registry.clear()
print(f"\nCleared registry: {list(munits.registry.keys())}")

# Now let's see what pandas does
import pandas.plotting._matplotlib.converter as converter

print("\n=== Testing pandas behavior on clean registry ===")
print(f"Before register: {list(munits.registry.keys())}")

converter.register()
print(f"After pandas register: {[str(k) for k in munits.registry.keys()]}")
print(f"_mpl_units cached: {[str(k) for k in converter._mpl_units.keys()]}")

converter.deregister()
print(f"After deregister: {[str(k) for k in munits.registry.keys()]}")

# The issue is clear: when registry starts empty, pandas doesn't cache anything
# But when deregistering, it restores from _mpl_units which might have been populated
# in a previous run

print("\n=== The problematic scenario ===")
munits.registry.clear()
converter._mpl_units.clear()

# Simulate matplotlib having already registered converters
# (This happens when matplotlib plots datetime data before pandas gets involved)
fig, ax = plt.subplots()
ax.plot([pydt.datetime(2020, 1, 1)], [1])  # This triggers matplotlib registration
plt.close('all')

print(f"After matplotlib plots datetime: {list(munits.registry.keys())}")
initial_keys = set(munits.registry.keys())

# Now pandas registers its converters
converter.register()
print(f"After pandas register: {list(munits.registry.keys())}")
print(f"Pandas cached converters: {list(converter._mpl_units.keys())}")

# Pandas deregisters
converter.deregister()
print(f"After pandas deregister: {list(munits.registry.keys())}")

final_keys = set(munits.registry.keys())
if initial_keys != final_keys:
    print(f"\nPROBLEM: Registry not restored to initial state!")
    print(f"  Initial: {initial_keys}")
    print(f"  Final: {final_keys}")
    print(f"  Extra keys: {final_keys - initial_keys}")
else:
    print("\nOK: Registry restored correctly")