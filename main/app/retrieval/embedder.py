"""Embedding utilities: sentence-transformers (e.g. all-MiniLM-L6-v2) or OpenAI API."""

from app.core.config import get_settings


def _get_embedder():
    s = get_settings()
    if (s.embedding_provider or "").lower() == "openai":
        return _OpenAIEmbedder(s)
    return _SentenceTransformerEmbedder(s)


class _SentenceTransformerEmbedder:
    def __init__(self, settings):
        self._settings = settings
        self._model = None

    _ST_DEFAULT = "all-MiniLM-L6-v2"

    def _model_load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            name = (self._settings.embedding_model_name or "").strip()
            if not name or "openai" in name.lower() or name.startswith("text-embedding"):
                name = self._ST_DEFAULT
            self._model = SentenceTransformer(name)
        return self._model

    def embed(self, text: str) -> list[float]:
        if not (text or "").strip():
            dim = self._settings.embedding_dim or 384
            return [0.0] * dim
        model = self._model_load()
        vec = model.encode(text, convert_to_numpy=True)
        return vec.tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.strip() if t else "" for t in texts]
        if all(not t for t in cleaned):
            dim = self._settings.embedding_dim or 384
            return [[0.0] * dim for _ in texts]
        model = self._model_load()
        vecs = model.encode(cleaned, convert_to_numpy=True)
        return [v.tolist() for v in vecs]


class _OpenAIEmbedder:
    def __init__(self, settings):
        self._settings = settings
        self._client = None
        self._model = (settings.embedding_model_name or "text-embedding-3-small").strip()

    def _client_get(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("openai package required for OpenAI embeddings: pip install openai")
            key = (self._settings.openai_api_key or "").strip()
            if not key:
                raise ValueError("OPENAI_API_KEY is required when embedding_provider=openai")
            self._client = OpenAI(api_key=key)
        return self._client

    def embed(self, text: str) -> list[float]:
        if not (text or "").strip():
            dim = self._settings.embedding_dim or 1536
            return [0.0] * dim
        client = self._client_get()
        r = client.embeddings.create(
            model=self._model,
            input=text.strip(),
        )
        return list(r.data[0].embedding)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.strip() if t else "" for t in texts]
        if all(not t for t in cleaned):
            dim = self._settings.embedding_dim or 1536
            return [[0.0] * dim for _ in texts]
        client = self._client_get()
        r = client.embeddings.create(
            model=self._model,
            input=cleaned,
        )
        by_idx = {d.index: list(d.embedding) for d in r.data}
        return [by_idx.get(i, [0.0] * (self._settings.embedding_dim or 1536)) for i in range(len(texts))]


def embed(text: str) -> list[float]:
    return _get_embedder().embed(text)


def embed_batch(texts: list[str]) -> list[list[float]]:
    return _get_embedder().embed_batch(texts)
