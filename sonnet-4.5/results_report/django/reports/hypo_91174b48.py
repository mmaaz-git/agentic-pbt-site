from io import StringIO
from django.core.serializers.base import ProgressBar
from hypothesis import given, strategies as st


@given(st.integers(min_value=1, max_value=1000))
def test_progressbar_handles_zero_total_count_gracefully(count):
    output = StringIO()
    pb = ProgressBar(output, total_count=0)

    pb.update(count)

# Run the test
if __name__ == "__main__":
    test_progressbar_handles_zero_total_count_gracefully()