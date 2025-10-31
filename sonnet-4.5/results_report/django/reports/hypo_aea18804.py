import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env/lib/python3.13/site-packages')

from hypothesis import given, strategies as st, settings
from django.core.exceptions import ImproperlyConfigured
from django.core.mail.backends.filebased import EmailBackend
from unittest.mock import patch, MagicMock


@given(st.none())
@settings(max_examples=10)
def test_filebased_backend_validates_path(none_value):
    mock_settings = MagicMock()
    del mock_settings.EMAIL_FILE_PATH

    with patch('django.core.mail.backends.filebased.settings', mock_settings):
        try:
            backend = EmailBackend(file_path=none_value)
            assert False, "Should have raised an exception"
        except ImproperlyConfigured:
            pass
        except TypeError:
            assert False, "Should raise ImproperlyConfigured, not TypeError"

if __name__ == "__main__":
    test_filebased_backend_validates_path()