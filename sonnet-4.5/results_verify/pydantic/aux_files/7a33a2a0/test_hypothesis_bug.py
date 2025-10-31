from hypothesis import given, strategies as st


@given(st.sampled_from(['true', 'True', 'TRUE', '1', '__all__']))
def test_disable_all_plugins_case_insensitive(value):
    disabled_plugins = value

    is_truthy = value.lower() in ('true', '1', '__all__')

    actual_disabled = disabled_plugins in ('__all__', '1', 'true')

    assert actual_disabled == is_truthy, (
        f"PYDANTIC_DISABLE_PLUGINS='{value}' should disable all plugins "
        f"regardless of case, but case-sensitive check fails"
    )

if __name__ == "__main__":
    test_disable_all_plugins_case_insensitive()