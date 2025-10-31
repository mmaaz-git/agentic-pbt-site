import matplotlib.units as munits
from pandas.plotting import register_matplotlib_converters, deregister_matplotlib_converters
from pandas.plotting._matplotlib.converter import register, deregister, _mpl_units
import datetime
import copy

print("Testing from pandas.plotting (public API):")
print("=" * 50)
initial = copy.copy(munits.registry)
print(f"Initial state - datetime.datetime in registry: {datetime.datetime in munits.registry}")

print("\nFirst cycle:")
register_matplotlib_converters()
print(f"After register: {datetime.datetime in munits.registry}")
deregister_matplotlib_converters()
print(f"After deregister: {datetime.datetime in munits.registry}")

print("\nSecond cycle:")
register_matplotlib_converters()
print(f"After register: {datetime.datetime in munits.registry}")
deregister_matplotlib_converters()
print(f"After deregister: {datetime.datetime in munits.registry}")

final = copy.copy(munits.registry)
print(f"\nRegistry restored properly? {initial == final}")

print("\n\nTesting from converter module directly:")
print("=" * 50)

# Clear state
munits.registry.clear()
_mpl_units.clear()

initial2 = copy.copy(munits.registry)
print(f"Initial state - datetime.datetime in registry: {datetime.datetime in munits.registry}")

print("\nFirst cycle:")
register()
print(f"After register: {datetime.datetime in munits.registry}")
print(f"_mpl_units state: {list(_mpl_units.keys())}")
deregister()
print(f"After deregister: {datetime.datetime in munits.registry}")
print(f"_mpl_units state: {list(_mpl_units.keys())}")

print("\nSecond cycle:")
register()
print(f"After register: {datetime.datetime in munits.registry}")
print(f"_mpl_units state: {list(_mpl_units.keys())}")
deregister()
print(f"After deregister: {datetime.datetime in munits.registry}")
print(f"_mpl_units state: {list(_mpl_units.keys())}")

final2 = copy.copy(munits.registry)
print(f"\nRegistry restored properly? {initial2 == final2}")

print("\n\nChecking _mpl_units accumulation:")
print("=" * 50)

# Clear state
munits.registry.clear()
_mpl_units.clear()

print("Cycle 1:")
print(f"Before: _mpl_units = {_mpl_units}")
register()
print(f"After register: _mpl_units = {_mpl_units}")
deregister()
print(f"After deregister: _mpl_units = {_mpl_units}")

print("\nCycle 2:")
register()
print(f"After register: _mpl_units keys = {list(_mpl_units.keys())}")
print(f"datetime.datetime converter type = {type(_mpl_units.get(datetime.datetime))}")
deregister()
print(f"After deregister: _mpl_units keys = {list(_mpl_units.keys())}")
print(f"datetime.datetime converter type = {type(_mpl_units.get(datetime.datetime))}")