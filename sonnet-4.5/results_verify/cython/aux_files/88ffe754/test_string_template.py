from string import Template

# Test if Python's string.Template mutates the input dict
tmpl = Template("Value: $x")
context = {'x': 42}

print("Before:", list(context.keys()))
result = tmpl.substitute(context)
print("After:", list(context.keys()))
print("Result:", result)