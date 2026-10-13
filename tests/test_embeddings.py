import math

from parking_assistant.embeddings import HashingEmbeddings, tokenize


def cosine(a, b):
    return sum(x * y for x, y in zip(a, b, strict=True))


def test_hashing_embeddings_are_deterministic_and_normalized():
    emb = HashingEmbeddings()
    first, second = (
        emb.embed_query("maximum vehicle height"),
        emb.embed_query("maximum vehicle height"),
    )
    assert first == second
    assert math.isclose(math.sqrt(sum(v * v for v in first)), 1.0)


def test_related_text_is_closer_than_unrelated():
    emb = HashingEmbeddings()
    query = emb.embed_query("How tall can my vehicle be? height limit")
    related = emb.embed_documents(["The maximum vehicle height is 2.1 metres."])[0]
    unrelated = emb.embed_documents(["Payment is made at the exit pay station."])[0]
    assert cosine(query, related) > cosine(query, unrelated)


def test_tokenize_removes_stopwords_and_stems():
    assert tokenize("Cancelling the reservations") == ["cancel", "reservation"]
