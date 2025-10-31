"""Check if list concatenation might be intentional for OpenAPI schema merging"""

from fastapi.utils import deep_dict_update
import json

# Test case that might be relevant for OpenAPI schemas
def test_openapi_tags_merging():
    """In OpenAPI, tags are often lists that might need to be combined"""
    print("Testing OpenAPI-like tag merging scenario:")

    main_schema = {
        "tags": ["user", "authentication"],
        "paths": {
            "/users": {
                "get": {
                    "tags": ["user"],
                    "summary": "Get users"
                }
            }
        }
    }

    additional_schema = {
        "tags": ["admin"],
        "paths": {
            "/users": {
                "get": {
                    "tags": ["admin"],
                    "security": [{"bearerAuth": []}]
                }
            }
        }
    }

    print("Main schema tags:", main_schema["tags"])
    print("Additional schema tags:", additional_schema["tags"])

    deep_dict_update(main_schema, additional_schema)
    print("After merge:", main_schema["tags"])
    print("Full merged schema:", json.dumps(main_schema, indent=2))
    print()

    # What if we apply the same update twice (e.g., in a retry scenario)?
    print("Applying the same update again...")
    deep_dict_update(main_schema, additional_schema)
    print("After second merge:", main_schema["tags"])
    print("Path tags:", main_schema["paths"]["/users"]["get"]["tags"])

def test_openapi_security_merging():
    """Security schemes in OpenAPI are also lists"""
    print("\nTesting OpenAPI security merging:")

    main = {
        "security": [
            {"apiKey": []},
            {"oauth2": ["read"]}
        ]
    }

    update = {
        "security": [
            {"bearerAuth": []}
        ]
    }

    print("Main security:", main["security"])
    print("Update security:", update["security"])

    deep_dict_update(main, update)
    print("After first merge:", main["security"])

    deep_dict_update(main, update)
    print("After second merge:", main["security"])
    print("Duplication occurred:", update["security"][0] in main["security"][1:])

def test_openapi_servers():
    """Servers in OpenAPI are lists"""
    print("\nTesting OpenAPI servers merging:")

    main = {
        "servers": [
            {"url": "https://api.example.com/v1"}
        ]
    }

    update = {
        "servers": [
            {"url": "https://staging.example.com/v1"}
        ]
    }

    print("Main servers:", main["servers"])
    print("Update servers:", update["servers"])

    deep_dict_update(main, update)
    print("After first merge:", main["servers"])

    deep_dict_update(main, update)
    print("After second merge:", main["servers"])
    print("Number of servers:", len(main["servers"]))

if __name__ == "__main__":
    test_openapi_tags_merging()
    test_openapi_security_merging()
    test_openapi_servers()