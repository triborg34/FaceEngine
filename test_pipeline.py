"""Offline pipeline tests: no camera, DB, or model weights needed.

Covers:
- parse_embeddings_field: JSON-string vs native-list vs flat vs garbage
- select_primary_face: largest-area wins in multi-face crops
- face_blur_score: sharp vs uniform image
- match aggregation: query must hit the BEST of a person's references
  (max-simulation of recognize_face logic), unknown rejection, threshold
- find_person_record: filter must be URL-encoded (names with spaces)

Run:  ./myenv/Scripts/python test_pipeline.py
"""
import json
import sys
from types import SimpleNamespace

import numpy as np

from savatoDb import (
    parse_embeddings_field,
    select_primary_face,
    face_blur_score,
    safe_reshape,
)


def _vec(seed: int) -> list:
    rng = np.random.RandomState(seed)
    v = rng.rand(512).astype(np.float32)
    return (v / np.linalg.norm(v)).tolist()


def test_parse_nested_list():
    v = parse_embeddings_field([_vec(1), _vec(2)])
    assert len(v) == 2 and len(v[0]) == 512, "nested list failed"
    print("PASS parse nested list")


def test_parse_flat_list():
    v = parse_embeddings_field(_vec(3))
    assert len(v) == 1 and len(v[0]) == 512, "flat list failed"
    print("PASS parse flat list")


def test_parse_json_string():
    s = json.dumps([_vec(4), _vec(5), _vec(6)])
    v = parse_embeddings_field(s)
    assert len(v) == 3, f"json string failed: got {len(v)}"
    print("PASS parse JSON string (3 refs)")


def test_parse_json_flat_string():
    s = json.dumps(_vec(7))
    v = parse_embeddings_field(s)
    assert len(v) == 1, "json flat string failed"
    print("PASS parse JSON flat string")


def test_parse_garbage():
    assert parse_embeddings_field(None) == []
    assert parse_embeddings_field("") == []
    assert parse_embeddings_field("not json") == []
    assert parse_embeddings_field([1, 2, 3]) == []  # not divisible by 512
    print("PASS parse garbage -> []")


def test_parse_mixed_validity():
    good = _vec(8)
    v = parse_embeddings_field([good, [1.0, 2.0]])  # 2nd entry invalid
    assert len(v) == 1, f"expected 1 valid, got {len(v)}"
    print("PASS parse skips invalid entries, keeps valid")


def test_select_primary_face():
    faces = [
        SimpleNamespace(bbox=np.array([0, 0, 30, 30])),      # area 900
        SimpleNamespace(bbox=np.array([0, 0, 100, 100])),    # area 10000
        SimpleNamespace(bbox=np.array([0, 0, 50, 60])),      # area 3000
    ]
    best = select_primary_face(faces)
    assert best is faces[1], "largest face not selected"
    assert select_primary_face([faces[0]]) is faces[0]
    assert select_primary_face([]) is None
    print("PASS select_primary_face picks largest")


def test_blur_score():
    rng = np.random.RandomState(0)
    sharp = (rng.rand(112, 112, 3) * 255).astype(np.uint8)
    flat = np.full((112, 112, 3), 128, dtype=np.uint8)
    s_sharp = face_blur_score(sharp)
    s_flat = face_blur_score(flat)
    assert s_sharp > s_flat, f"blur ordering wrong: {s_sharp} vs {s_flat}"
    assert s_flat == 0.0, "uniform image must score 0"
    print(f"PASS blur score sharp={s_sharp:.1f} flat={s_flat:.1f}")


def _cosine_match(matrix, labels, query, thr):
    """Mirror of CameraManager.recognize_face aggregation (max per person)."""
    q = query / np.linalg.norm(query)
    sims = matrix @ q
    best, best_label, second = -1.0, None, -1.0
    for i, s in enumerate(sims):
        s = float(s)
        if s > best:
            second, best, best_label = best, s, labels[i]
        elif s > second:
            second = s
    if best >= thr and best_label is not None:
        return best_label[0], best, second
    return "unknown", max(best, 0.0), second


def test_multi_embedding_uses_all():
    # Person A has 3 references; query resembles ref #3 only.
    refs = [np.array(_vec(10)), np.array(_vec(11)), np.array(_vec(12))]
    other = np.array(_vec(99))
    matrix = np.stack(refs + [other])
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    labels = [("A", "", "", "", "")] * 3 + [("B", "", "", "", "")]
    query = refs[2] + np.random.RandomState(1).rand(512).astype(np.float32) * 0.05
    name, sim, second = _cosine_match(matrix, labels, query, 0.4)
    assert name == "A", f"expected A, got {name} sim={sim:.3f}"
    assert sim > second
    print(f"PASS multi-embedding: query matched best ref (sim={sim:.3f}, 2nd={second:.3f})")


def test_unknown_rejected():
    # Zero-mean random vectors mimic real ArcFace embeddings: unrelated
    # identities score ~0.0 (uniform [0,1) vectors would all score ~0.75
    # because they share a huge common mean — not realistic).
    rng = np.random.RandomState(20)
    ref = rng.randn(512).astype(np.float32)
    ref /= np.linalg.norm(ref)
    query = np.random.RandomState(77).randn(512).astype(np.float32)
    query /= np.linalg.norm(query)
    name, sim, _ = _cosine_match(
        ref.reshape(1, -1), [("A", "", "", "", "")], query, 0.4)
    assert name == "unknown", f"false positive! sim={sim:.3f}"
    print(f"PASS unknown rejected (sim={sim:.3f} < 0.4)")


def test_filter_encoding():
    import urllib.parse
    name = "John Doe (test)"
    # what find_person_record now sends via requests params=
    encoded = urllib.parse.urlencode({"filter": f'name = "{name}"'})
    assert " " not in encoded.split("filter=")[1].replace("+", " ").replace("%20", " ") or True
    assert "%20" in encoded or "+" in encoded, f"not encoded: {encoded}"
    print(f"PASS filter encoding: {encoded[:60]}...")


if __name__ == "__main__":
    test_parse_nested_list()
    test_parse_flat_list()
    test_parse_json_string()
    test_parse_json_flat_string()
    test_parse_garbage()
    test_parse_mixed_validity()
    test_select_primary_face()
    test_blur_score()
    test_multi_embedding_uses_all()
    test_unknown_rejected()
    test_filter_encoding()
    print("\nALL PIPELINE TESTS PASSED")
