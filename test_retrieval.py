"""Unit tests for retrieval query expansion (no ML deps)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from retrieval import (
    expand_queries,
    is_coverage_question,
    preferred_doc_names,
    _topic_from_coverage_question,
    _topic_words,
)


def test_wear_and_tear_queries():
    q = "Is wear and tear covered under the home contents policy?"
    assert is_coverage_question(q)
    expanded = expand_queries(q)
    assert any("Exclusion" in e for e in expanded)
    assert _topic_words(q) == ["wear", "tear"]


def test_domestic_animals_queries():
    q = "Does the policy cover damage caused by domestic animals I own?"
    assert is_coverage_question(q)
    topic = _topic_from_coverage_question(q)
    assert "domestic animal" in topic.lower()
    assert any("Exclusion" in e for e in expand_queries(q))


def test_temp_accommodation_queries():
    q = "What is the maximum period for temporary accommodation cover?"
    expanded = expand_queries(q)
    assert any("Event 21" in e for e in expanded)


def test_wear_tear_matches_not_pay_heading():
    from ingestor import Chunk

    chunk = Chunk(
        chunk_id="x",
        doc_name="qbe_home_policy_wording_mil.pdf",
        page=4,
        section="Section 1",
        heading_path="Section 1 - Home Contents > What We cover > 1. We will not pay",
        clause_ref="",
        text="caused by Wear, Tear; However We will pay if any other Event",
    )
    from retrieval import keyword_exclusion_supplement

    hits = keyword_exclusion_supplement(
        [chunk],
        "Is wear and tear covered under the home contents policy?",
        None,
    )
    assert len(hits) == 1


def test_prefer_home_policy():
    docs = [
        "qbe_home_policy_wording_mil.pdf",
        "QM8698-1124 QBE Contents Insurance PDS.pdf",
    ]
    pref = preferred_doc_names(
        "Does the policy cover damage caused by domestic animals I own?", docs
    )
    assert pref == ["qbe_home_policy_wording_mil.pdf"]


if __name__ == "__main__":
    test_wear_and_tear_queries()
    test_domestic_animals_queries()
    test_temp_accommodation_queries()
    test_prefer_home_policy()
    print("All retrieval unit tests passed.")
