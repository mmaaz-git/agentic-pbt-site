import django
from django.conf import settings

settings.configure(
    DEBUG=True,
    SECRET_KEY='test-key',
    DATABASES={'default': {'ENGINE': 'django.db.backends.sqlite3', 'NAME': ':memory:'}},
    INSTALLED_APPS=['django.contrib.contenttypes', 'django.contrib.auth'],
)
django.setup()

from hypothesis import given, strategies as st, settings as hypo_settings
from django.views.generic.edit import ModelFormMixin, DeletionMixin

@st.composite
def url_template_with_placeholders(draw):
    num_placeholders = draw(st.integers(min_value=1, max_value=5))
    placeholders = [draw(st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier()))
                    for _ in range(num_placeholders)]
    template_parts = ['/redirect']
    for placeholder in placeholders:
        template_parts.append(f'/{{{placeholder}}}')
    template_parts.append('/')
    return ''.join(template_parts)

@given(
    url_template_with_placeholders(),
    st.dictionaries(
        st.text(min_size=1, max_size=20).filter(lambda x: x.isidentifier()),
        st.one_of(st.text(min_size=0, max_size=100), st.integers(), st.none()),
        min_size=1, max_size=10
    )
)
@hypo_settings(max_examples=500)
def test_success_url_format_with_object_dict(url_template, object_dict):
    class MockObject:
        def __init__(self, attrs):
            self.__dict__.update(attrs)

    class TestModelFormMixin(ModelFormMixin):
        success_url = url_template

    view = TestModelFormMixin()
    view.object = MockObject(object_dict)

    try:
        url = view.get_success_url()
        print(f"✓ Success URL generated: {url}")
    except KeyError as e:
        print(f"✗ KeyError raised for template '{url_template}' with object_dict {object_dict}: {e}")
        raise

if __name__ == "__main__":
    print("Running Hypothesis test for ModelFormMixin.get_success_url()")
    print("=" * 60)
    try:
        test_success_url_format_with_object_dict()
        print("\nAll tests passed!")
    except Exception as e:
        print(f"\nTest failed!")
        import traceback
        traceback.print_exc()