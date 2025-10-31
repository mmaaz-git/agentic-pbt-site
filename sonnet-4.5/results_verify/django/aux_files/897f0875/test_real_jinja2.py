import jinja2

# Test what a real Jinja2 TemplateSyntaxError looks like
try:
    env = jinja2.Environment()
    template = env.from_string("{{ invalid syntax }}")
except jinja2.TemplateSyntaxError as e:
    print("Real Jinja2 TemplateSyntaxError attributes:")
    print(f"  lineno: {e.lineno}")
    print(f"  filename: {e.filename}")
    print(f"  message: {e.message}")
    print(f"  source: {repr(e.source)}")
    print(f"  type(lineno): {type(e.lineno)}")

# Test with another syntax error
try:
    env = jinja2.Environment()
    template = env.from_string("{% for item in %}")
except jinja2.TemplateSyntaxError as e:
    print("\nAnother real Jinja2 TemplateSyntaxError:")
    print(f"  lineno: {e.lineno}")
    print(f"  filename: {e.filename}")
    print(f"  message: {e.message}")
    print(f"  source: {repr(e.source)}")

# Test with a template from file
import tempfile
import os

with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
    f.write("line1\nline2\n{% invalid %}\nline4")
    temp_path = f.name

try:
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(os.path.dirname(temp_path)))
    template = env.get_template(os.path.basename(temp_path))
except jinja2.TemplateSyntaxError as e:
    print(f"\nFile-based TemplateSyntaxError:")
    print(f"  lineno: {e.lineno}")
    print(f"  filename: {e.filename}")
    print(f"  message: {e.message}")
    print(f"  source: {repr(e.source)}")
finally:
    os.unlink(temp_path)

# Check if lineno can ever be None or negative in real Jinja2
print("\nChecking Jinja2 source expectations...")
print(f"Jinja2 version: {jinja2.__version__}")