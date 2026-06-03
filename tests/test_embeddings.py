from regurag.embeddings.encoder import l2_normalize, with_prefix


def test_with_prefix_only_changes_text_when_prefix_exists() -> None:
    assert with_prefix("", "hello") == "hello"
    assert with_prefix("query: ", "hello") == "query: hello"


def test_l2_normalize() -> None:
    assert l2_normalize([3.0, 4.0]) == [0.6, 0.8]
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]
