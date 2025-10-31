import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/xarray_env/lib/python3.13/site-packages')

import xarray.core.formatting_html as fmt_html
import inspect

test_html = "<div>Test Content</div>"

result_default = fmt_html._wrap_datatree_repr(test_html)
result_false = fmt_html._wrap_datatree_repr(test_html, end=False)
result_true = fmt_html._wrap_datatree_repr(test_html, end=True)

sig = inspect.signature(fmt_html._wrap_datatree_repr)
actual_default = sig.parameters['end'].default

print(f"Actual default value in signature: {actual_default}")
print(f"Documented default (from docstring): True")
print()
print(f"Default behavior matches end=False: {result_default == result_false}")
print(f"Default behavior matches end=True:  {result_default == result_true}")