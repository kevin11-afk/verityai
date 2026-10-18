"""Extractive answer composition: grounded, sentence-level, no fragments or headings."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from api.services.answer import clean_markdown, compose_answer  # noqa: E402

DOC = """# Generalization and bias

## Why detectors struggle with new generators

Detectors trained on one family of generators often fail badly on a different family, even
when both families look equally fake to people. This happens because classifiers learn
generator-specific artifacts.

- Accuracy on a held-out set from the same generators should not be read as accuracy on a new one.
- Another unrelated bullet about cooking pasta with plenty of salty boiling water at home.

Source: Ojha, U. (2023). Towards Universal Fake Image Detectors. https://arxiv.org/abs/2302.10174
"""


def P(text, i=1):
    return SimpleNamespace(text=text, citation_id=i)


def test_claims_are_verbatim_sentences_from_the_cited_passage():
    claims = compose_answer("Why do detectors fail on new generators?", [P(DOC)])
    assert claims
    for text, cid in claims:
        assert cid == 1 and text in clean_markdown(DOC)
        assert text[0].isupper() and text.rstrip().endswith((".", "!", "?"))


def test_headings_citations_and_urls_are_never_claims():
    claims = [t for t, _ in compose_answer("generators bias Ojha Universal Fake Image Detectors", [P(DOC)], max_claims=5)]
    assert not any(t.startswith("Generalization and bias") or "Why detectors struggle" == t for t in claims)
    assert not any(t.lower().startswith("source:") or "arxiv.org" in t for t in claims)


def test_mid_sentence_fragments_from_chunk_boundaries_are_skipped():
    fragment = "same generators used in training should not be read as the accuracy you would get later. A complete sentence about generators is here, and it is long enough."
    claims = [t for t, _ in compose_answer("generators accuracy training", [P(fragment)])]
    assert claims and all(t[0].isupper() for t in claims)
    assert not any(t.startswith("same generators") for t in claims)


def test_one_claim_per_passage_and_ordered_by_relevance():
    a = P("Content Credentials are signed metadata attached to media when it is captured.", 1)
    b = P("Cats and dogs are common household pets that people keep indoors for companionship.", 2)
    c = P("Provenance metadata can be stripped when files are re-encoded or screenshotted by users.", 3)
    claims = compose_answer("What are content credentials and provenance metadata?", [a, b, c])
    assert [cid for _, cid in claims] and len({cid for _, cid in claims}) == len(claims)
    assert 2 not in {cid for _, cid in claims}


def test_no_overlap_means_no_claims_rather_than_a_guess():
    assert compose_answer("zzqx plumbing xylophone", [P(DOC)]) == []
    assert compose_answer("the of and", [P(DOC)]) == []


def test_light_stemming_matches_plural_and_verb_forms():
    claims = compose_answer("detector failing", [P("A detector fails when generators change between releases of models.")])
    assert claims


def test_wrapped_list_items_are_kept_whole():
    doc = ("- **License:** Apache-2.0, fully permissive including commercial use,\n"
           "  modification, and redistribution of the detector weights.\n"
           "- Short.\n")
    claims = [t for t, _ in compose_answer("detector license commercial use redistribution", [P(doc)])]
    assert claims == ["License: Apache-2.0, fully permissive including commercial use, modification, and redistribution of the detector weights."]


def test_abbreviations_do_not_split_sentences():
    text = "Classifiers trained to detect one family of generators (e.g. GANs) often fail on a different family, as shown by Corvi et al. in 2023."
    claims = [t for t, _ in compose_answer("classifiers generators family fail", [P(text)])]
    assert claims == [text]


def test_source_lines_merged_by_chunk_overlap_are_not_claims():
    text = "Content Credentials record how media was made. Source: Coalition for Content Provenance and Authenticity (C2PA)."
    claims = [t for t, _ in compose_answer("content credentials provenance authenticity", [P(text)], max_claims=5)]
    assert all(not t.lower().startswith("source:") for t in claims) and claims


def test_sentences_with_dangling_pronouns_rank_below_self_contained_ones():
    a = P("It contains real videos alongside four kinds of manipulations across compression levels.", 1)
    b = P("FaceForensics++ contains real videos alongside four kinds of manipulations across compression levels.", 2)
    claims = compose_answer("FaceForensics++ videos manipulations", [a, b], max_claims=2)
    assert claims[0][1] == 2
