from langgraph.graph import StateGraph, END

from .states import GraphState
from .nodes import (
    router_node,
    decide_route,
    retrieve_vector_node,
    web_search_node,
    llm_fallback_node,
    generate_node,
    critique_govern_node,
    decide_repair,
)


def build_aura_app():
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("router", router_node)
    graph.add_node("retrieve_vector", retrieve_vector_node)
    graph.add_node("web_search", web_search_node)
    graph.add_node("llm_fallback", llm_fallback_node)
    graph.add_node("generate", generate_node)
    graph.add_node("critique_govern", critique_govern_node)

    # Entry point: router
    graph.set_entry_point("router")

    # Route based on router decision
    graph.add_conditional_edges(
        "router",
        decide_route,
        {
            "vector": "retrieve_vector",
            "web": "web_search",
            "fallback": "llm_fallback",
        },
    )

    # Retrieval → generation
    graph.add_edge("retrieve_vector", "generate")
    graph.add_edge("web_search", "generate")

    # Fallback goes straight to critique/govern
    graph.add_edge("llm_fallback", "critique_govern")

    # Generated answer → critique/govern
    graph.add_edge("generate", "critique_govern")

    # Repair loop from critique/govern
    graph.add_conditional_edges(
        "critique_govern",
        decide_repair,
        {
            "accept": END,
            "retry_search": "web_search",      # try web search instead
            "retry_generate": "generate",      # re-generate with same docs
            "fallback": "llm_fallback",        # pure LLM
        },
    )

    app = graph.compile()
    return app


# Convenience singleton
app = build_aura_app()
