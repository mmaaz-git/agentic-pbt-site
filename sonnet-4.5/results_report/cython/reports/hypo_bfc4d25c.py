import tempfile
import shutil
from hypothesis import given, strategies as st, settings
from Cython.Debugger.DebugWriter import CythonDebugWriter

# Define valid XML tag name strategies
valid_xml_start = st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_')
valid_xml_chars = st.sampled_from('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.')
valid_xml_tag = st.builds(
    lambda start, rest: start + rest,
    valid_xml_start,
    st.text(alphabet=valid_xml_chars, max_size=20)
)

@given(valid_xml_tag, st.dictionaries(valid_xml_tag, st.text()))
@settings(max_examples=100)
def test_add_entry_with_attrs(tag_name, attrs):
    tmpdir = tempfile.mkdtemp()
    try:
        writer = CythonDebugWriter(tmpdir)
        writer.tb.start('Module', {})
        writer.add_entry(tag_name, **attrs)
        writer.tb.end('Module')
        writer.tb.end('cython_debug')
        writer.tb.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

if __name__ == "__main__":
    # Run the test
    test_add_entry_with_attrs()