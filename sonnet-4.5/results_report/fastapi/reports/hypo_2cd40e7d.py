from hypothesis import given, strategies as st
from fastapi.openapi.utils import get_openapi

@given(
    title=st.text(min_size=1, max_size=100),
    version=st.text(min_size=1, max_size=50),
    servers=st.one_of(st.none(), st.just([]), st.lists(st.fixed_dictionaries({"url": st.just("http://localhost")})))
)
def test_get_openapi_servers_preservation(title, version, servers):
    result = get_openapi(title=title, version=version, servers=servers, routes=[])

    if servers is not None:
        assert "servers" in result, f"servers field should be present when servers={servers}"
        assert result["servers"] == servers, f"servers value should be {servers} but got {result.get('servers')}"


@given(
    title=st.text(min_size=1, max_size=100),
    version=st.text(min_size=1, max_size=50),
    tags=st.one_of(st.none(), st.just([]), st.lists(st.fixed_dictionaries({"name": st.text(min_size=1, max_size=50)})))
)
def test_get_openapi_tags_preservation(title, version, tags):
    result = get_openapi(title=title, version=version, tags=tags, routes=[])

    if tags is not None:
        assert "tags" in result, f"tags field should be present when tags={tags}"
        assert result["tags"] == tags, f"tags value should be {tags} but got {result.get('tags')}"


if __name__ == "__main__":
    print("Testing servers preservation...")
    test_get_openapi_servers_preservation()
    print()

    print("Testing tags preservation...")
    test_get_openapi_tags_preservation()
    print()

    print("All tests passed if no output above!")