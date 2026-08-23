from app.services.rag_service import SAMPLE_DOCUMENTS, keyword_retrieve


def test_keyword_retrieve_finds_phishing_docs():
    results = keyword_retrieve("what is phishing?", SAMPLE_DOCUMENTS)
    assert results
    blob = " ".join(doc["text"] for doc in results).lower()
    assert "phishing" in blob
    assert results[0]["metadata"]["topic"] == "phishing"


def test_keyword_retrieve_finds_rate_limiting():
    results = keyword_retrieve("What is rate limiting?", SAMPLE_DOCUMENTS)
    assert results
    assert "rate limiting" in results[0]["text"].lower()


def test_keyword_retrieve_ignores_unrelated_chat():
    results = keyword_retrieve("Hey how's things going?", SAMPLE_DOCUMENTS)
    assert results == []
