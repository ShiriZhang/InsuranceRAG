from scripts.select_clause_v2 import CachingEmbedder


class RecordingEmbedder:
    model_id = "fixture-model"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[float(len(text))] for text in texts]


def test_caching_embedder_reuses_vectors_across_budget_trials():
    underlying = RecordingEmbedder()
    embedder = CachingEmbedder(underlying, "fixture-model")

    first = embedder.embed_texts(["alpha", "beta", "alpha"])
    second = embedder.embed_texts(["beta", "gamma"])

    assert embedder.model_id == "fixture-model"
    assert first == [[5.0], [4.0], [5.0]]
    assert second == [[4.0], [5.0]]
    assert underlying.calls == [["alpha", "beta"], ["gamma"]]


def test_caching_embedder_reuses_vectors_across_processes(tmp_path):
    cache_path = tmp_path / "embeddings.jsonl"
    first_underlying = RecordingEmbedder()
    first = CachingEmbedder(
        first_underlying, "fixture-model", cache_path
    )
    assert first.embed_texts(["alpha"]) == [[5.0]]

    second_underlying = RecordingEmbedder()
    second = CachingEmbedder(
        second_underlying, "fixture-model", cache_path
    )
    assert second.embed_texts(["alpha", "beta"]) == [[5.0], [4.0]]
    assert second_underlying.calls == [["beta"]]
