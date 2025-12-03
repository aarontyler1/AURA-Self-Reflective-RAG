from typing import List, Dict, Any, Literal
from typing_extensions import TypedDict


Route = Literal["vector", "web", "fallback"]
Decision = Literal["accept", "retry_search", "retry_generate", "fallback"]


class GraphState(TypedDict, total=False):
    """
    Shared state passed between nodes in the A.U.R.A. graph.
    """
    question: str
    route: Route
    documents: List[str]          # retrieved context
    generation: str               # model answer
    critique: Dict[str, Any]      # raw critique info
    decision: Decision            # governance decision
