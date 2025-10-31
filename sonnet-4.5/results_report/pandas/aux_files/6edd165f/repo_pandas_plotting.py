import matplotlib.units as munits
from pandas.plotting import register_matplotlib_converters, deregister_matplotlib_converters
import datetime

print("Initial state:")
print(f"datetime.datetime in registry: {datetime.datetime in munits.registry}")

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