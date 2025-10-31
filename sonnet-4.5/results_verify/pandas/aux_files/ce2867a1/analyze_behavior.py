import matplotlib.units as munits
import pandas.plotting._matplotlib.converter as converter
import datetime as pydt
import numpy as np

print("=== Initial state analysis ===")
print(f"Initial registry: {list(munits.registry.keys())}")
print(f"Initial _mpl_units: {converter._mpl_units}")

print("\n=== After register() ===")
converter.register()
print(f"Registry after register: {list(munits.registry.keys())}")
print(f"_mpl_units after register: {list(converter._mpl_units.keys())}")

# Check if the datetime converters were already there
for key in [pydt.datetime, pydt.date, np.datetime64]:
    if key in converter._mpl_units:
        print(f"  {key}: cached converter type = {type(converter._mpl_units[key])}")

print("\n=== After deregister() ===")
converter.deregister()
print(f"Registry after deregister: {list(munits.registry.keys())}")
print(f"_mpl_units still contains: {list(converter._mpl_units.keys())}")

# Let's see what's happening in detail
print("\n=== Detailed flow analysis ===")
munits.registry.clear()
converter._mpl_units.clear()

print("Starting from empty registry")
print(f"Registry: {list(munits.registry.keys())}")

# Now register
print("\nCalling register()...")
converter.register()

print(f"After register:")
print(f"  Registry keys: {[str(k) for k in munits.registry.keys()]}")
print(f"  _mpl_units keys: {[str(k) for k in converter._mpl_units.keys()]}")

# Check what was cached
for type_, cls in converter.get_pairs():
    if type_ in converter._mpl_units:
        print(f"  Cached {type_}: {converter._mpl_units[type_]}")

print("\nCalling deregister()...")
converter.deregister()

print(f"After deregister:")
print(f"  Registry keys: {[str(k) for k in munits.registry.keys()]}")
print(f"  What got restored: {[(str(k), type(v)) for k, v in munits.registry.items()]}")

# Test the deregister logic manually
print("\n=== Manual deregister logic test ===")
munits.registry.clear()
converter._mpl_units.clear()

# Simulate what happens
print("Simulating register with some pre-existing converters...")

# First, let's say matplotlib already has some converters
class MockConverter:
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"MockConverter({self.name})"

# Pre-populate with matplotlib's own converters (simulated)
munits.registry[pydt.datetime] = MockConverter("matplotlib-datetime")
munits.registry[pydt.date] = MockConverter("matplotlib-date")
munits.registry[np.datetime64] = MockConverter("matplotlib-datetime64")

print(f"Initial registry (with matplotlib converters): {list(munits.registry.items())[:3]}")

# Now pandas registers
converter.register()
print(f"After pandas register: {len(munits.registry)} converters")
print(f"Cached in _mpl_units: {list(converter._mpl_units.keys())}")

# Now deregister
converter.deregister()
print(f"After deregister: {len(munits.registry)} converters")
print(f"Converters in registry: {list(munits.registry.keys())}")