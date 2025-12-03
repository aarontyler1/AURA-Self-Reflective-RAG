from pprint import pprint

from .graph import app


def run_aura(question: str, verbose: bool = True):
    """
    Stream state updates through the A.U.R.A. LangGraph app and
    return the final generation and critique.
    """
    inputs = {"question": question}
    final_state: dict = {}

    for update in app.stream(inputs):
        for node_name, state in update.items():
            # Merge this node's update into the full state
            final_state.update(state)

            print(f"\n=== Node: {node_name} ===")
            for k in ["route", "documents", "generation", "decision"]:
                if k in state:
                    snippet = state[k]
                    if isinstance(snippet, str) and len(snippet) > 300:
                        snippet = snippet[:300] + "..."
                    print(f"{k}: {snippet}")


    return final_state


if __name__ == "__main__":
    q = "How does A.U.R.A. improve reliability in research workflows?"
    result = run_aura(q, verbose=True)

    print("\n\n=== Final Answer ===")
    print(result.get("generation", "No answer produced."))

    print("\n=== Critique ===")
    pprint(result.get("critique", {}))
