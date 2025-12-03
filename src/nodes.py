from __future__ import annotations
import numpy as np
import json
import math
from typing import Dict, Any, List

import requests

from .config import OLLAMA_BASE_URL, CHAT_MODEL, EMBED_MODEL
from .states import GraphState, Route, Decision


# This section relates to retrieval from internal vector DB for this A.U.R.A Implementation.
# --------------------------------------------------------------------
# Simple in-memory "company data" corpus for demo purposes
# To be replaced in practice with company documents / vector DB.
# --------------------------------------------------------------------
CORPUS: List[str] = [
    "A.U.R.A. is a self-reflective RAG framework that combines query routing, retrieval, "
    "generation, and governance to support research workflows.",
    "Retrieval-augmented generation (RAG) improves reliability by grounding model outputs "
    "in a document store or knowledge base.",
    "Agentic AI systems coordinate multiple specialised components, such as routers, "
    "retrievers and governance agents, to achieve complex goals.",
]

# Pre-computed embeddings cache (simple optimisation)
_CORPUS_EMBEDDINGS: List[np.ndarray] | None = None


# --------------------------------------------------------------------
# Utility helpers
# --------------------------------------------------------------------
def embed_texts(texts: List[str]) -> List[np.ndarray]:
    """
    Get embeddings for a list of texts using Ollama's embedding API.
    We embed one text at a time for simplicity.
    """
    embeddings: List[np.ndarray] = []
    for t in texts:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": t},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        vec = np.array(data["embedding"], dtype="float32")
        embeddings.append(vec)
    return embeddings

def ollama_chat(system_prompt: str, user_prompt: str) -> str:
    """
    Call a local LLaMA model via Ollama's chat API.
    """
    payload = {
        "model": CHAT_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
    }

    resp = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    # Response structure: {"message": {"role": "assistant", "content": "..."}}
    return data["message"]["content"].strip()


# Best used for example case where vector embeddings are not present with Corpus.
def get_corpus_embeddings() -> List[np.ndarray]:
    global _CORPUS_EMBEDDINGS
    if _CORPUS_EMBEDDINGS is None:
        _CORPUS_EMBEDDINGS = embed_texts(CORPUS)
    return _CORPUS_EMBEDDINGS

# Important to recognise the cosine similarity scores between the question/query vector and the document vectors.
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# --------------------------------------------------------------------
# Query Layer (router)
# --------------------------------------------------------------------
def router_node(state: GraphState) -> GraphState:
    """
    First layer which decides whether to use internal vector data, web search, or fallback.
    For MVP (Minimum Viable Products) we use simple heuristics; you can swap this for an LLM-based router under fallback
    decisions.
    """
    question = state["question"].lower()

    # Very basic heuristic routing example to help present idea.
    if any(word in question for word in ["today", "current", "latest", "news", "2024"]):
        route: Route = "web"
    elif len(question.split()) < 4:
        # Very vague queries -> let base model try first to determine likelihood of LLM response
        route = "fallback"
    else:
        route = "vector"

    return {"route": route}


# Router for LangGraphs conditional edges.
def decide_route(state: GraphState) -> str:
    return state.get("route", "vector")


# --------------------------------------------------------------------
# Retrieval: vector store
# --------------------------------------------------------------------
def retrieve_vector_node(state: GraphState) -> GraphState:
    """
    Retrieve documents from the in-memory corpus using embeddings.
    """
    question = state["question"]
    q_emb = embed_texts([question])[0]
    corpus_embs = get_corpus_embeddings()

    scored = [
        (cosine_similarity(q_emb, emb), doc)
        for emb, doc in zip(corpus_embs, CORPUS)
    ]
    scored.sort(reverse=True, key=lambda x: x[0])
    top_docs = [doc for _, doc in scored[:3]]
    # top_docs decides the top k scored documents to keep (set to 3 in this case)

    return {"documents": top_docs}


# --------------------------------------------------------------------
# Retrieval: web search (Not currently functioning fully (psuedo case))
# --------------------------------------------------------------------
def web_search_node(state: GraphState) -> GraphState:
    """
    Placeholder web search node.

    For now, we simulate web search by returning an empty document list and
    letting the generator rely on the base model. Later I will explore the implmentation of some web tools like Tavily or
    SerpAPI here.
    """
    question = state["question"]
    pseudo_doc = (
        "Web search placeholder. The model should answer using its general knowledge. "
        f"User question: {question}"
    )
    return {"documents": [pseudo_doc]}


# --------------------------------------------------------------------
# Fallback: pure LLM generation
# --------------------------------------------------------------------
def llm_fallback_node(state: GraphState) -> GraphState:
    """
    Answer using the base LLaMA model without any retrieval, only based on trained knowledge.
    """
    question = state["question"]

    system_prompt = (
        "You are A.U.R.A., an AI assistant. Answer concisely, honestly and to the best of your ability. "
        "If you are uncertain, it is very important to say so explicitly rather than produce an incorrect response."
    )

    answer = ollama_chat(system_prompt, question)
    return {"generation": answer}


# --------------------------------------------------------------------
# Generation section: Combines the question + the retrieved docs
# Feeds them to LLaMA with instructions to use the context
# --------------------------------------------------------------------
def generate_node(state: GraphState) -> GraphState:
    """
    Generate an answer using retrieved documents for your context/evidence (RAG-style approach).
    If no documents are present, behaves like LLM-only generation.
    """
    question = state["question"]
    docs = state.get("documents", [])

    if docs:
        context_block = "\n\n---\n\n".join(docs)
    else:
        context_block = "No external documents were retrieved."

    system_prompt = (
        "You are A.U.R.A., a helpful research assistant. Use the provided context where "
        "possible, but do not fabricate citations. If the context is unrelated to the question, "
        "answer based on your own knowledge and say that the answer is not fully grounded."
    )

    user_prompt = (
        f"User question:\n{question}\n\n"
        f"Context documents:\n{context_block}\n\n"
        "Give a clear, concise answer in a few paragraphs."
    )

    answer = ollama_chat(system_prompt, user_prompt)
    return {"generation": answer}



# --------------------------------------------------------------------
# Critique + Governance (combined node framework)
# --------------------------------------------------------------------
def critique_govern_node(state: GraphState) -> GraphState:
    """
    Self-critique step using LLaMA via Ollama:
    - Is the answer grounded in the documents?
    - Does it address the question?
    Produces a decision used by the repair loop.
    """
    question = state["question"]
    answer = state.get("generation", "")
    docs = state.get("documents", [])
    context_excerpt = "\n\n---\n\n".join(docs[:3]) if docs else "No documents."

    critique_prompt = f"""
You are a strict governance agent reviewing a model answer.

User question:
{question}

Context documents:
{context_excerpt}

Model answer:
{answer}

Reply in valid JSON with the following fields:
- "grounded": true or false  (is the answer supported by the documents?)
- "answers_question": true or false (does it actually answer the user's question?)
- "notes": short natural language explanation.
"""

    system_prompt = "Return ONLY valid JSON. Do not add commentary."

    raw = ollama_chat(system_prompt, critique_prompt)

    try:
        data: Dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "grounded": False,
            "answers_question": False,
            "notes": f"Failed to parse JSON critique. Raw output: {raw[:200]}",
        }

    grounded = bool(data.get("grounded", False))
    answers = bool(data.get("answers_question", False))

    if grounded and answers:
        decision: Decision = "accept"
    elif not grounded:
        decision = "retry_search"
    elif grounded and not answers:
        decision = "retry_generate"
    else:
        decision = "fallback"

    return {
        "critique": data,
        "decision": decision,
    }



# Conditional edge helper for LangGraph
def decide_repair(state: GraphState) -> str:
    return state.get("decision", "accept")
