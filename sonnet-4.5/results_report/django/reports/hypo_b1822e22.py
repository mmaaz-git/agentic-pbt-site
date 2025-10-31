from hypothesis import given, strategies as st
from django.core.checks.registry import CheckRegistry
from django.core.checks import Info


@given(st.text(min_size=1), st.text(min_size=1))
def test_double_registration_preserves_both_tags(tag1, tag2):
    from hypothesis import assume
    assume(tag1 != tag2)

    registry = CheckRegistry()

    def my_check(app_configs=None, **kwargs):
        return [Info("test")]

    registry.register(my_check, tag1)
    registry.register(my_check, tag2)

    available_tags = registry.tags_available()

    assert tag1 in available_tags, f"Tag '{tag1}' should still be available"
    assert tag2 in available_tags, f"Tag '{tag2}' should be available"


if __name__ == "__main__":
    # Run the test with Hypothesis
    test_double_registration_preserves_both_tags()