from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Optional, List
from guardrails import check_guardrails
from extract_filters import exract_filters_from_user_query
from qdrant_search import ProductSearch
from qdrant_client import models


class AgentState(TypedDict):
    user_query: str
    image_base64: Optional[str]
    guardrails_passed: bool
    error_message: Optional[str]
    filters: Optional[str]
    search_results: Optional[List[str]]


def verify_guardrails(state: AgentState) -> AgentState:
    """Check if query relates to allowed product categories."""
    user_query = state["user_query"]
    image_base64 = state.get("image_base64")

    # Skip guardrails if image is provided (either image-only or image + text)
    if image_base64:
        state["guardrails_passed"] = True
        return state

    # Only check guardrails for text-only queries
    if user_query and user_query.strip():
        result = check_guardrails(user_query, provider="groq")
        state["guardrails_passed"] = result.lower().strip() == "true"

        if not state["guardrails_passed"]:
            state["error_message"] = (
                "Sorry, I can only help with bathroom, kitchen, lighting, "
                "furniture, and audio products."
            )
    else:
        # No query and no image - should not happen due to validation in API
        state["guardrails_passed"] = False
        state["error_message"] = "Please provide a search query or image."

    return state


def extract_filters(state: AgentState) -> AgentState:
    """Extract filters from text query."""
    user_query = state["user_query"]

    if user_query and user_query.strip():
        try:
            state["filters"] = exract_filters_from_user_query(user_query)
        except Exception as e:
            print(f"Filter extraction error: {e}")
            state["filters"] = None
    else:
        state["filters"] = None

    return state


def perform_search(state: AgentState) -> AgentState:
    """Execute product search with optional filters."""
    user_query = state["user_query"]
    image_base64 = state.get("image_base64")
    filters_str = state.get("filters")

    try:
        search = ProductSearch()
        qdrant_filters = None

        # Parse filter string to Qdrant Filter object
        if filters_str and filters_str.strip():
            try:
                # Clean markdown code blocks
                clean = filters_str.strip()
                if clean.startswith("```python"):
                    clean = clean[10:].strip()
                elif clean.startswith("```"):
                    clean = clean[3:].strip()
                if clean.endswith("```"):
                    clean = clean[:-3].strip()

                # Execute filter code (LLM generates without assignment)
                scope = {"models": models}
                exec(f"filters = {clean}", scope)
                qdrant_filters = scope.get("filters")
            except Exception as e:
                print(f"Filter parsing error: {e}")

        # Perform search based on input type
        if image_base64 and user_query:
            # Image similarity + text filters
            results = search.search_by_image_base64(
                base64_image=image_base64,
                limit=100,
                filters=qdrant_filters
            )
        elif image_base64:
            # Image similarity only
            results = search.search_by_image_base64(
                base64_image=image_base64,
                limit=100,
                filters=None
            )
        else:
            # Text similarity + filters
            results = search.search_by_text(
                query_text=user_query,
                limit=100,
                filters=qdrant_filters
            )

        # Extract SKU IDs
        state["search_results"] = [
            r.get("payload", {}).get("SKU") or r.get("payload", {}).get("sku")
            for r in results
            if isinstance(r, dict) and (r.get("payload", {}).get("SKU") or r.get("payload", {}).get("sku"))
        ]

    except Exception as e:
        print(f"Search error: {e}")
        state["search_results"] = []
        state["error_message"] = f"Search error: {str(e)}"

    return state


def route_after_guardrails(state: AgentState) -> str:
    """Route based on guardrails result."""
    return "extract_filters" if state["guardrails_passed"] else END


# Build graph
graph = StateGraph(AgentState)
graph.add_node("verify_guardrails", verify_guardrails)
graph.add_node("extract_filters", extract_filters)
graph.add_node("perform_search", perform_search)
graph.add_edge(START, "verify_guardrails")
graph.add_conditional_edges(
    "verify_guardrails",
    route_after_guardrails,
    {"extract_filters": "extract_filters", END: END}
)
graph.add_edge("extract_filters", "perform_search")
graph.add_edge("perform_search", END)

app = graph.compile()


if __name__ == "__main__":
    # Sample base64 image for testing
    base64_image = """
data:image/jpeg;base64,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
    """
    test_cases = [
        {
            "name": "Example 1: Text Search with Filters",
            "state": {
                "user_query": "faucet in matte black from Riobel under $2000",
                "image_base64": None,
                "guardrails_passed": False,
                "error_message": None,
                "filters": None,
                "search_results": None
            }
        },
        {
            "name": "Example 2: Text + Image Search",
            "state": {
                "user_query": "Riobel brand in matte black under $500",
                "image_base64": base64_image,
                "guardrails_passed": False,
                "error_message": None,
                "filters": None,
                "search_results": None
            }
        },
        {
            "name": "Example 3: Image Only Search",
            "state": {
                "user_query": "",
                "image_base64": base64_image,
                "guardrails_passed": False,
                "error_message": None,
                "filters": None,
                "search_results": None
            }
        },
        {
            "name": "Example 4: Invalid Query",
            "state": {
                "user_query": "Show me laptops for gaming",
                "image_base64": None,
                "guardrails_passed": False,
                "error_message": None,
                "filters": None,
                "search_results": None
            }
        }
    ]

    for test in test_cases:
        print(f"\n{'='*60}")
        print(f"{test['name']}")
        print('='*60)
        result = app.invoke(test["state"])

        if result.get("error_message"):
            print(f"❌ {result['error_message']}")
        else:
            count = len(result.get("search_results", []))
            print(f"✅ Found {count} product(s)")
            if count > 0 and count <= 100:
                print(f"   SKUs: {result['search_results']}")
