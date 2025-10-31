"""Code from the bug report to reproduce the issue"""

import os
from flask.helpers import get_debug_flag, get_load_dotenv

os.environ["FLASK_DEBUG"] = " false "
result = get_debug_flag()
print(f"FLASK_DEBUG=' false ' returns: {result}")
assert result is True

os.environ["FLASK_DEBUG"] = " 0 "
result = get_debug_flag()
print(f"FLASK_DEBUG=' 0 ' returns: {result}")
assert result is True

os.environ["FLASK_SKIP_DOTENV"] = " false "
result = get_load_dotenv()
print(f"FLASK_SKIP_DOTENV=' false ' returns: {result}")
assert result is False

os.environ["FLASK_SKIP_DOTENV"] = " 0 "
result = get_load_dotenv()
print(f"FLASK_SKIP_DOTENV=' 0 ' returns: {result}")
assert result is False

print("All assertions passed - bug confirmed!")