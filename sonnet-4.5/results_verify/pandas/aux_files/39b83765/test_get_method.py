from pandas.plotting._misc import _Options

opts = _Options()

# Check if get method is overridden
print(f"'get' in _Options.__dict__: {'get' in _Options.__dict__}")
print(f"'get' method is from dict: {_Options.get is dict.get}")

# Show that other methods are overridden
print(f"\n'__getitem__' in _Options.__dict__: {'__getitem__' in _Options.__dict__}")
print(f"'__setitem__' in _Options.__dict__: {'__setitem__' in _Options.__dict__}")
print(f"'__contains__' in _Options.__dict__: {'__contains__' in _Options.__dict__}")
print(f"'__delitem__' in _Options.__dict__: {'__delitem__' in _Options.__dict__}")

# Test the behavior
opts["xaxis.compat"] = True
print(f"\nopts['xaxis.compat'] = {opts['xaxis.compat']}")  # Uses overridden __getitem__
print(f"opts['x_compat'] = {opts['x_compat']}")  # Uses overridden __getitem__ with alias

print(f"\nopts.get('xaxis.compat') = {opts.get('xaxis.compat')}")  # Uses inherited dict.get
print(f"opts.get('x_compat') = {opts.get('x_compat')}")  # Uses inherited dict.get - doesn't handle alias

print(f"\n'x_compat' in opts = {'x_compat' in opts}")  # Uses overridden __contains__
print(f"'xaxis.compat' in opts = {'xaxis.compat' in opts}")  # Uses overridden __contains__