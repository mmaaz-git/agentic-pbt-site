from hypothesis import given, strategies as st
import jinja2
from starlette.templating import Jinja2Templates

@given(st.sampled_from([[], (), ""]))
def test_jinja2templates_empty_directory_with_env(empty_directory):
    custom_env = jinja2.Environment()

    templates = Jinja2Templates(directory=empty_directory, env=custom_env)

    assert templates.env is custom_env, \
        f"Expected templates.env to be custom_env when directory={empty_directory!r}, but got a different env"

if __name__ == "__main__":
    test_jinja2templates_empty_directory_with_env()