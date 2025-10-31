from hypothesis import given, strategies as st
from django.core.files.base import File

class FalsyButOpenFile:
    def __bool__(self):
        return False

    @property
    def closed(self):
        return False

@given(st.just(None))
def test_closed_property_with_falsy_file(x):
    falsy_file = FalsyButOpenFile()
    file_obj = File(falsy_file)

    assert not falsy_file.closed
    assert not file_obj.closed

# Run the test
test_closed_property_with_falsy_file()