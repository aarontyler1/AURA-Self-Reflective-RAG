import os

# Ollama runs locally at this address by default
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Chat + embedding models to use
CHAT_MODEL = os.getenv("AURA_CHAT_MODEL", "llama3.2:3b")
EMBED_MODEL = os.getenv("AURA_EMBED_MODEL", "nomic-embed-text")
