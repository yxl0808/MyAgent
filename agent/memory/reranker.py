"""Lazy local cross-encoder; weights are downloaded on first use."""
import threading
from typing import List


class LocalReranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", device="cpu",
                 batch_size=4, max_length=1024):
        self.model_name = model_name
        self.device = device
        self.batch_size = max(1, batch_size)
        self.max_length = max_length
        self._model = None
        self._load_error = None
        self._lock = threading.Lock()

    def __call__(self, query: str, documents: List[str]) -> List[float]:
        if not documents:
            return []
        # Serialize loading and inference for shared managers in Web sessions.
        with self._lock:
            if self._load_error is not None:
                raise RuntimeError('Reranker load failed; restart after fixing dependencies/model path') from self._load_error
            if self._model is None:
                try:
                    from sentence_transformers import CrossEncoder
                    self._model = CrossEncoder(self.model_name, device=self.device,
                                               max_length=self.max_length, trust_remote_code=False)
                except Exception as exc:
                    self._load_error = exc
                    raise
            values = self._model.predict([(query, text) for text in documents],
                                         batch_size=self.batch_size,
                                         show_progress_bar=False)
            return [float(value) for value in values]
