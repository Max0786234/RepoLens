from repolens.languages import detect_language, is_indexable, is_secret_path, is_test_path
from repolens.text import split_identifier, stem, tokenize, tokenize_query


def test_split_identifier_handles_snake_and_camel_case():
    assert split_identifier("getHTTPResponse_code") == ["get", "http", "response", "code"]
    assert split_identifier("parse_header_value") == ["parse", "header", "value"]


def test_tokenize_keeps_full_identifier_and_parts():
    tokens = tokenize("def should_strip_auth(self):")
    assert "should_strip_auth" in tokens
    assert {"strip", "auth"} <= set(tokens)


def test_stemming_matches_plurals_and_ing():
    assert stem("retries") == "retry"
    assert stem("headers") == "header"
    assert stem("redirecting") == "redirect"
    assert stem("class") == "class"


def test_query_stopwords_are_removed():
    assert tokenize_query("Where is the retry logic implemented?") == ["retry"]
    # A query made only of stopwords still produces tokens.
    assert tokenize_query("how does this work") != []


def test_language_detection_and_filters():
    assert detect_language("src/app.py") == "python"
    assert detect_language("lib/Thing.KT") == "kotlin"
    assert detect_language("Dockerfile") == "dockerfile"
    assert detect_language("photo.png") is None
    # v1 bug: the app accepted these suffixes but the retriever never indexed them.
    for path in ["a.php", "b.rb", "c.cs", "d.swift", "e.kt", "f.c", "g.sql", "h.sh"]:
        assert is_indexable(path), path


def test_ignored_and_secret_paths():
    assert not is_indexable("node_modules/x/index.js")
    assert not is_indexable("dist/bundle.min.js")
    assert not is_indexable("package-lock.json")
    assert is_secret_path(".env") and is_secret_path("config/.env.production") and is_secret_path("certs/server.pem")
    assert not is_indexable(".env")


def test_test_path_detection():
    assert is_test_path("tests/test_cart.py")
    assert is_test_path("src/foo_test.go")
    assert is_test_path("web/app.spec.ts")
    assert not is_test_path("src/testing_utils_core.py")
