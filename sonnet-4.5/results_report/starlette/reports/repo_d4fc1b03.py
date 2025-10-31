import jinja2
from starlette.templating import Jinja2Templates

custom_env = jinja2.Environment()

templates = Jinja2Templates(directory=[], env=custom_env)

print(f"templates.env is custom_env: {templates.env is custom_env}")

assert templates.env is custom_env, "Expected custom_env to be used, but a new env was created instead"