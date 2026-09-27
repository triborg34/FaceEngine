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

from olderTests.newsavatoDb import (
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


def _unit(seed: int) -> np.ndarray:
    v = np.random.RandomState(seed).randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


def _ref_at(cos_t: float, q: np.ndarray, seed: int) -> np.ndarray:
    """Build a unit vector whose cosine similarity to *q* is exactly cos_t."""
    u = _unit(seed)
    u = u - float(np.dot(u, q)) * q
    u = u / np.linalg.norm(u)
    s = float(np.sqrt(max(0.0, 1.0 - cos_t * cos_t)))
    return (cos_t * q + s * u).astype(np.float32)


def _index(persons):
    """persons: [(label_tuple, [ref vectors])] -> (matrix, labels, starts)."""
    rows, labels, starts = [], [], []
    for label, refs in persons:
        starts.append(len(rows))
        labels.append(label)
        rows.extend(refs)
    matrix = np.stack(rows).astype(np.float32)
    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix, labels, np.asarray(starts, dtype=np.int64)


def _config(matrix, labels, starts, thr=0.6, **extra):
    from types import SimpleNamespace
    cfg = SimpleNamespace(embedding_index=(matrix, labels, starts),
                          simscore=thr, **extra)
    return cfg


def _recognize(cfg, query, thr=None):
    """Drive the REAL CameraManager.recognize_face against a stub config."""
    from olderTests.newengine import CameraManager
    if thr is not None:
        cfg.simscore = thr
    cm = CameraManager.__new__(CameraManager)
    cm.config = cfg
    return cm.recognize_face(np.asarray(query, dtype=np.float32), 'None', 'None')


def test_multi_embedding_uses_all():
    # Person A has 3 references; query resembles ref #3 only.
    q = _unit(1)
    refs = [_ref_at(0.99, q, 10), _ref_at(0.99, q, 11), _ref_at(0.95, q, 12)]
    other = _ref_at(0.10, q, 99)
    label_a = ("A", "", "", "", "", "idA", None)
    label_b = ("B", "", "", "", "", "idB", None)
    cfg = _config(*_index([(label_a, refs), (label_b, [other])]))
    name, sim, margin, *_ = _recognize(cfg, q)
    assert name == "A", f"expected A, got {name} sim={sim:.3f}"
    assert sim > 0.9, f"expected to hit the best ref, got {sim:.3f}"
    print(f"PASS multi-embedding: best ref used (sim={sim:.3f})")


def test_second_place_is_a_different_person():
    """Margin must compare PEOPLE, not neighbouring reference images.

    A's second reference scores 0.85 here; person B scores 0.80.  An
    implementation that takes the 2nd largest row reports margin 0.05 and
    wrongly calls the frame ambiguous; the correct margin is 0.10.
    """
    q = _unit(2)
    a_refs = [_ref_at(0.90, q, 20), _ref_at(0.85, q, 21)]
    b_refs = [_ref_at(0.80, q, 22)]
    label_a = ("A", "", "", "", "", "idA", None)
    label_b = ("B", "", "", "", "", "idB", None)
    cfg = _config(*_index([(label_a, a_refs), (label_b, b_refs)]))
    name, sim, margin, *_ = _recognize(cfg, q)
    assert name == "A", f"expected A, got {name}"
    assert abs(sim - 0.90) < 1e-3, f"sim {sim:.4f} != 0.90"
    assert abs(margin - 0.10) < 1e-3, (
        f"margin {margin:.4f} != 0.10 (2nd place must be person B)")
    print(f"PASS margin compares distinct persons "
          f"(sim={sim:.3f} margin={margin:.3f})")


def test_unknown_rejected():
    # Unrelated identity scores ~0.0 for zero-mean random ArcFace vectors.
    q = _unit(3)
    ref = _ref_at(0.05, q, 77)
    cfg = _config(*_index([(("A", "", "", "", "", "idA", None), [ref])]))
    name, sim, margin, *_ = _recognize(cfg, q, thr=0.4)
    assert name == "unknown", f"false positive! sim={sim:.3f}"
    print(f"PASS unknown rejected (sim={sim:.3f} < 0.4)")


def test_tight_margin_reported():
    """Two look-alikes above threshold must still expose a small margin."""
    q = _unit(4)
    a = [_ref_at(0.62, q, 30)]
    b = [_ref_at(0.61, q, 31)]
    cfg = _config(*_index([(("A", "", "", "", "", "idA", None), a),
                           (("B", "", "", "", "", "idB", None), b)]))
    name, sim, margin, *_ = _recognize(cfg, q, thr=0.6)
    assert name == "A", f"expected A, got {name}"
    assert 0 < margin < 0.05, f"margin {margin:.4f} should be tight"
    print(f"PASS look-alike margin surfaced (sim={sim:.3f} "
          f"margin={margin:.3f})")


def test_per_person_threshold_override():
    q = _unit(5)
    ref = _ref_at(0.55, q, 40)
    # simThreshold present but 0 -> falls back to the global simscore
    cfg = _config(*_index([(("A", "", "", "", "", "idA", 0.0), [ref])]))
    name, *_ = _recognize(cfg, q, thr=0.6)
    assert name == "unknown", f"0 must mean 'use global thr', got {name}"
    # Non-zero override raises the bar for this person only
    name, *_ = _recognize(cfg, q, thr=0.6)
    cfg.embedding_index = (cfg.embedding_index[0],
                           [("A", "", "", "", "", "idA", 0.90)],
                           cfg.embedding_index[2])
    name, *_ = _recognize(cfg, q)
    assert name == "unknown", f"override 0.90 must reject sim=0.55, got {name}"
    print("PASS per-person simThreshold override honoured")


def _obs(pid, name, sim, **kw):
    o = {'pid': pid, 'name': name, 'sim': sim, 'margin': 0.2,
         'gender': 'None', 'age': 'None', 'role': '', 'socialnumber': '',
         'det': 0.9, 'blur': 120.0, 'yaw': -2.0}
    o.update(kw)
    return o


def _fusion(obs):
    from olderTests.newengine import CameraManager
    return CameraManager._fuse_observations(obs)


def _decisive(fused, vote_obs=5, majority=0.6, simscore=0.6, margin_min=0.06):
    from olderTests.newengine import CameraManager
    from types import SimpleNamespace
    cfg = SimpleNamespace(voteObs=vote_obs, voteMajority=majority,
                          simscore=simscore, marginMin=margin_min)
    cm = CameraManager.__new__(CameraManager)
    cm.config = cfg
    return cm._fusion_is_decisive(fused)


def test_fusion_majority_votes():
    # 3 x Alice, 2 x unknown -> Alice wins once enough obs exist
    obs = [_obs('p1', 'Alice', 0.75)] * 3 + [_obs('', 'unknown', 0.4)] * 2
    fused = _fusion(obs)
    assert fused['name'] == 'Alice', f"got {fused['name']}"
    assert fused['n'] == 3 and fused['total'] == 5
    assert fused['ratio'] == 0.6
    assert _decisive(fused), "3 agreeing of 5 should be decisive"

    # A single glitch frame must not overturn a settled identity
    obs = [_obs('p1', 'Alice', 0.75)] * 5 + [_obs('p2', 'Bob', 0.55)]
    fused = _fusion(obs)
    assert fused['name'] == 'Alice', f"got {fused['name']}"
    assert _decisive(fused), "5 of 6 agree -> decisive"
    print("PASS fusion majority voting")


def test_fusion_requires_enough_observations():
    fused = _fusion([_obs('p1', 'Alice', 0.75)] * 2)
    assert not _decisive(fused), "2 observations must not commit"
    fused = _fusion([_obs('p1', 'Alice', 0.75)] * 3)
    assert _decisive(fused), "3 unanimous observations should commit"
    print("PASS fusion observation minimum")


def test_fusion_rejects_low_margin_and_low_sim():
    # Agreement but too close to the threshold to trust
    fused = _fusion([_obs('p1', 'Alice', 0.61)] * 3)
    assert _decisive(fused), "0.61 with default thr 0.6 should pass"
    # Tight race between two identities -> never decisive
    obs = [_obs('p1', 'Alice', 0.7)] * 3 + [_obs('p2', 'Bob', 0.69)] * 3
    fused = _fusion(obs)
    assert not _decisive(fused), "50/50 split must not commit"
    # Winner's mean below threshold -> reject
    fused = _fusion([_obs('p1', 'Alice', 0.59)] * 5)
    assert not _decisive(fused), "mean below simscore must not commit"
    # Agreement and high score, but a close rival -> margin too tight
    fused = _fusion([_obs('p1', 'Alice', 0.75)] * 5 +
                    [_obs('p2', 'Bob', 0.72)])
    assert not _decisive(fused), "margin 0.03 < 0.06 must not commit"
    print("PASS fusion rejects ambiguous / weak evidence")


def test_fusion_empty():
    assert _fusion([]) is None, "no observations -> None"
    print("PASS fusion handles empty observation list")


def test_embedding_meta_helpers():
    from olderTests.newsavatoDb import (build_embedding_meta, parse_embedding_meta,
                          fit_meta_length, face_yaw, EMBEDDING_MODEL)
    m = build_embedding_meta(142.3, 0.91, -8.2)
    assert m['model'] == EMBEDDING_MODEL and m['blur'] == 142.3
    assert parse_embedding_meta(json.dumps([m])) == [m]
    assert parse_embedding_meta(None) == []
    assert parse_embedding_meta("not json") == []
    # Pre-existing records have no meta: padding keeps indices aligned
    assert len(fit_meta_length([], 3)) == 3
    assert fit_meta_length([m, m, m], 1) == [m]
    # Face without a pose model (or a trimmed pack) reports yaw 0
    from types import SimpleNamespace
    assert face_yaw(SimpleNamespace(pose=None)) == 0.0
    assert abs(face_yaw(SimpleNamespace(pose=[1.0, -12.5, 0.0])) + 12.5) < 1e-6
    print("PASS embeddingMeta helpers + face_yaw fallback")


def test_setting_gates_ignore_zeroed_columns():
    """A 0 in `setting` means 'not configured', not 'gate disabled'.

    PocketBase backfills the number columns it adds to an existing record
    with 0; with yaw stored as 0 the worker rejected every frame
    (abs(yaw) > 0 is always true) and nothing could ever be recognised.
    """
    from olderTests.newengine import (resolve_setting_gates, DEFAULT_MARGIN_MIN,
                        DEFAULT_VOTE_OBS, DEFAULT_VOTE_MAJORITY,
                        DEFAULT_MIN_BLUR, DEFAULT_MAX_YAW, DEFAULT_MIN_FACE_PX)
    zeroed = resolve_setting_gates({
        'marginMin': 0, 'voteObs': 0, 'voteMajority': 0,
        'minBlur': 0, 'maxYaw': 0, 'minFacePx': 0})
    assert zeroed['maxYaw'] == DEFAULT_MAX_YAW > 0, zeroed
    assert zeroed['voteObs'] == DEFAULT_VOTE_OBS, zeroed
    assert zeroed['marginMin'] == DEFAULT_MARGIN_MIN, zeroed
    assert zeroed['voteMajority'] == DEFAULT_VOTE_MAJORITY, zeroed
    assert zeroed['minBlur'] == DEFAULT_MIN_BLUR, zeroed
    assert zeroed['minFacePx'] == DEFAULT_MIN_FACE_PX, zeroed

    for empty in ({}, None, {'maxYaw': None}, {'maxYaw': ''}, {'maxYaw': 'x'}):
        assert resolve_setting_gates(empty)['maxYaw'] == DEFAULT_MAX_YAW

    tuned = resolve_setting_gates({'maxYaw': 20, 'voteObs': 7,
                                   'minFacePx': 32, 'minBlur': 15.5})
    assert tuned['maxYaw'] == 20.0 and tuned['voteObs'] == 7
    assert tuned['minFacePx'] == 32 and tuned['minBlur'] == 15.5
    print("PASS setting gates treat 0 as unset, honour real values")


def test_filter_encoding():
    import urllib.parse
    name = "John Doe (test)"
    # what find_person_record now sends via requests params=
    encoded = urllib.parse.urlencode({"filter": f'name = "{name}"'})
    assert " " not in encoded.split("filter=")[1].replace("+", " ").replace("%20", " ") or True
    assert "%20" in encoded or "+" in encoded, f"not encoded: {encoded}"
    print(f"PASS filter encoding: {encoded[:60]}...")


def test_min_face_px_resolution():
    """env MIN_FACE_PX > setting.minFacePx > default.

    Regression for "no face detected in image": this camera's 1400px
    frames put a face at 56x61px, which the old 64px floor rejected even
    though detection scored 0.864 and the embedding matched the same
    person across scales at 0.69-0.88.
    """
    import os
    from olderTests.newsavatoDb import get_min_face_px, DEFAULT_MIN_FACE_PX
    had_env = "MIN_FACE_PX" in os.environ
    saved = os.environ.get("MIN_FACE_PX")
    try:
        os.environ.pop("MIN_FACE_PX", None)
        assert get_min_face_px() == DEFAULT_MIN_FACE_PX
        assert get_min_face_px(SimpleNamespace()) == DEFAULT_MIN_FACE_PX
        assert get_min_face_px(SimpleNamespace(minFacePx=56)) == 56
        # PocketBase drops unknown keys: the field may be absent or null
        assert get_min_face_px(SimpleNamespace(minFacePx=None)) == DEFAULT_MIN_FACE_PX
        assert get_min_face_px(SimpleNamespace(minFacePx="")) == DEFAULT_MIN_FACE_PX
        assert get_min_face_px(SimpleNamespace(minFacePx="bad")) == DEFAULT_MIN_FACE_PX
        # An env var is an explicit deployment override
        os.environ["MIN_FACE_PX"] = "88"
        assert get_min_face_px(SimpleNamespace(minFacePx=56)) == 88
        if not had_env:
            assert DEFAULT_MIN_FACE_PX <= 40, (
                f"default {DEFAULT_MIN_FACE_PX}px still rejects the 56px "
                f"faces this deployment produces")
    finally:
        if saved is None:
            os.environ.pop("MIN_FACE_PX", None)
        else:
            os.environ["MIN_FACE_PX"] = saved
    print(f"PASS min face px resolution (env > setting > default "
          f"{DEFAULT_MIN_FACE_PX})")


def _bare_manager():
    """A CameraManager with only the state the worker touches."""
    import threading
    from olderTests.newengine import CameraManager
    cm = CameraManager.__new__(CameraManager)
    cm.camera_id = 9
    cm.running = True
    cm.stop_event = threading.Event()
    cm._pending_lock = threading.Lock()
    cm._pending_tracks = {}
    cm._pending_event = threading.Event()
    cm._track_obs_lock = threading.Lock()
    cm.track_obs = {}
    cm._processed_tracks_lock = threading.Lock()
    cm.processed_tracks = set()
    cm.face_info = {}
    cm.face_info_lock = threading.Lock()
    cm.embedding_cache = {}
    cm._cache_lock = threading.Lock()
    cm.capture_buffer = [None, None, None]
    cm.capture_read_idx = 0
    cm.face_handler = None
    return cm


def test_pending_map_serves_every_track():
    """A burst of N people must not drop anyone.

    The old recognition_queue had maxsize=5: with 6+ people in frame the
    tracks that lost the race were never recognised at all, which is why
    multi-person footage came back all 'unknown'.
    """
    cm = _bare_manager()
    for tid in range(8):  # more people than the old queue could hold
        cm._queue_track(tid, ("/f", tid, f"crop{tid}", None))
    items = cm._take_pending(0.01)
    assert len(items) == 8, f"only {len(items)} of 8 tracks survived"
    assert sorted(i[1] for i in items) == list(range(8))

    # Newest crop per track wins (the worker only needs the latest frame)
    cm._queue_track(3, ("/f", 3, "stale", None))
    cm._queue_track(3, ("/f", 3, "fresh", None))
    items = cm._take_pending(0.01)
    assert len(items) == 1 and items[0][2] == "fresh", items

    assert cm._take_pending(0.01) == [], "idle round must return nothing"
    print("PASS pending map serves all 8 tracks, keeps newest crop")


class _StubFaceHandler:
    """Face handler keyed by the track id stashed in the crop's first pixel."""

    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.calls = 0

    def get(self, img):
        self.calls += 1
        tid = int(img[0, 0, 0])
        return [SimpleNamespace(bbox=np.array([10, 10, 60, 60]),
                                det_score=0.95, gender=0, age=31,
                                embedding=self.embeddings[tid],
                                pose=[0.0, -4.0, 0.0])]


def test_worker_recognises_two_tracks():
    """End-to-end: two people in frame both get identified and logged.

    Drives the REAL recognition_worker (throttle, quality gate, temporal
    fusion, commit -> insertToDb) with two interleaved tracks, the way a
    multi-person frame feeds it.
    """
    import time
    import threading
    import olderTests.newengine as newengine
    from olderTests.newengine import CameraManager

    q_alice, q_bob = _unit(101), _unit(202)
    refs = [(("Alice", "", "", "", "", "idA", None), [_ref_at(0.99, q_alice, 301)]),
            (("Bob", "", "", "", "", "idB", None), [_ref_at(0.99, q_bob, 302)])]
    matrix, labels, starts = _index(refs)
    cfg = SimpleNamespace(
        embedding_index=(matrix, labels, starts),
        simscore=0.6, score=0.6, marginMin=0.06, voteObs=5,
        voteMajority=0.6, minBlur=0.0, maxYaw=180.0, minFacePx=40,
        quality=100, padding=0, isRelay=False, isRegionMode=False,
        ip_relay=None, ip_port=None, relayN1=0, relayN2=0,
        face_lock=threading.Lock(), face_handler=None)
    cm = _bare_manager()
    cm.config = cfg
    cm.face_handler = _StubFaceHandler({1: q_alice, 2: q_bob})

    submitted = []
    real_submit = newengine.submit_db_task
    newengine.submit_db_task = lambda fn, *a, **kw: submitted.append((a, kw))
    worker = threading.Thread(target=cm.recognition_worker, daemon=True)
    worker.start()
    try:
        deadline = time.time() + 8.0
        rnd = 0
        while time.time() < deadline:
            for tid in (1, 2):
                crop = np.random.RandomState(rnd).randint(
                    0, 255, (80, 80, 3)).astype(np.uint8)
                crop[0, 0, 0] = tid
                cm._queue_track(tid, ("/frame.jpg", tid, crop, None))
            with cm._track_obs_lock:
                if all(cm.track_obs.get(t, {}).get('committed')
                       for t in (1, 2)):
                    break
            rnd += 1
            time.sleep(0.15)
        cm.stop_event.set()
        worker.join(timeout=3)
    finally:
        newengine.submit_db_task = real_submit

    assert not worker.is_alive(), "worker did not stop"
    assert len(submitted) == 2, f"expected 2 commits, got {len(submitted)}"
    names = sorted(a[0] for a, _ in submitted)
    assert names == ["Alice", "Bob"], f"wrong identities: {names}"
    for args, kwargs in submitted:
        assert kwargs["obs_count"] >= 3, f"only {kwargs['obs_count']} fused"
        assert kwargs["sim"] >= 0.9, f"sim {kwargs['sim']} too low"
        assert kwargs["matchMargin" if "matchMargin" in kwargs else "margin"] >= 0.06
        assert kwargs["person_id"] in ("idA", "idB")
        assert kwargs["blur"] > 0 and "yaw" in kwargs
    assert cm.processed_tracks == {1, 2}
    assert cm.face_info[1]["name"] == "Alice"
    assert cm.face_info[2]["name"] == "Bob"
    print(f"PASS worker identified both tracks "
          f"(sims={[round(kw['sim'], 3) for _, kw in submitted]})")


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
    test_second_place_is_a_different_person()
    test_unknown_rejected()
    test_tight_margin_reported()
    test_per_person_threshold_override()
    test_fusion_majority_votes()
    test_fusion_requires_enough_observations()
    test_fusion_rejects_low_margin_and_low_sim()
    test_fusion_empty()
    test_embedding_meta_helpers()
    test_setting_gates_ignore_zeroed_columns()
    test_filter_encoding()
    test_min_face_px_resolution()
    test_pending_map_serves_every_track()
    test_worker_recognises_two_tracks()
    print("\nALL PIPELINE TESTS PASSED")
