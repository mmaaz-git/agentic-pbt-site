import sys
sys.path.insert(0, '/home/npc/pbt/agentic-pbt/envs/django_env')

from hypothesis import given, strategies as st, settings
from django.core.management.templates import TemplateCommand

@given(st.sampled_from(["http", "https", "ftp"]))
@settings(max_examples=100)
def test_is_url_rejects_protocol_only_urls(protocol):
    cmd = TemplateCommand()
    protocol_only_url = f"{protocol}://"

    if cmd.is_url(protocol_only_url):
        tmp = protocol_only_url.rstrip("/")
        filename = tmp.split("/")[-1]

        assert ":" not in filename, \
            f"is_url({protocol_only_url!r}) returns True but produces invalid filename: {filename!r}"

        assert len(filename) > 0, \
            f"is_url({protocol_only_url!r}) returns True but produces empty filename"

# Run the test
if __name__ == "__main__":
    test_is_url_rejects_protocol_only_urls()