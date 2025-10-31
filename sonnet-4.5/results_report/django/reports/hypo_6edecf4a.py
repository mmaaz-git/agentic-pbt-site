import django
from django.conf import settings

if not settings.configured:
    settings.configure(DEBUG=True, SECRET_KEY='test')
    django.setup()

from django.forms.utils import ErrorDict, ErrorList
from django.forms.renderers import get_default_renderer
from hypothesis import given, strategies as st

@given(
    st.dictionaries(
        st.text(min_size=1, max_size=50),
        st.lists(st.text(min_size=1, max_size=100), min_size=1, max_size=5).map(
            lambda errors: ErrorList(errors)
        ),
        min_size=0,
        max_size=10
    )
)
def test_errordict_copy_preserves_renderer(error_data):
    renderer = get_default_renderer()

    error_dict = ErrorDict(error_data, renderer=renderer)

    copied = error_dict.copy()

    assert hasattr(copied, 'renderer'), "Copy should have renderer attribute"
    assert copied.renderer == error_dict.renderer, "Copy should preserve renderer"

if __name__ == "__main__":
    test_errordict_copy_preserves_renderer()