from pandas.plotting._misc import _Options

# Create an _Options instance
opts = _Options()

# Set a value using the canonical key
opts["xaxis.compat"] = True

# Try to get the value using both canonical key and alias
result_canonical = opts.get("xaxis.compat", "default_value")
result_alias = opts.get("x_compat", "default_value")

print(f"opts.get('xaxis.compat', 'default_value') = {result_canonical}")
print(f"opts.get('x_compat', 'default_value') = {result_alias}")

# For comparison, show that __getitem__ works correctly with the alias
print(f"opts['x_compat'] = {opts['x_compat']}")

# Show that __contains__ also works correctly with the alias
print(f"'x_compat' in opts = {'x_compat' in opts}")