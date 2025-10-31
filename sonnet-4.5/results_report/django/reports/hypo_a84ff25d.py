from hypothesis import given, strategies as st, assume
from django.core.serializers.base import ProgressBar
from io import StringIO

@given(st.integers(min_value=0, max_value=1000), st.integers(min_value=0, max_value=1000))
def test_progress_bar_no_crash(total_count, count):
    assume(count <= total_count)
    output = StringIO()
    pb = ProgressBar(output, total_count)
    pb.update(count)

if __name__ == "__main__":
    test_progress_bar_no_crash()