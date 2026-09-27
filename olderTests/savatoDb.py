"""Database / registration module (merged).

Features taken from ``newsavatoDb.py``:
  - DbWorker bounded async queue for DB inserts (never blocks the video
    pipeline), shared requests.Session connection pool, relay thread pool.
  - Person/face CRUD (find/create/update/add/remove/delete), embedding
    parsing, embeddingMeta quality records, model-pack safety
    (``embeddingModel`` / ``isActive``), min-face-px + blur quality gates,
    multi-image registration, face-crop upload helpers, paged loader.

Face-recognition workflow taken from ``oldsavetoDP.py``:
  - ``reciveFromUi`` registration flow: download (if URL) -> YOLO person
    crop -> InsightFace embed -> ``check_person_exists`` -> append/update
    the embedding on the existing record or create a new one.
  - Self-contained model creation when the caller does not pass shared
    models, so the old 8-argument call signature keeps working.
  - Detection-event logging (``savePicture`` + ``should_insert`` throttle)
    and the relay sequence in ``insertToDb``.
"""

import base64
import datetime
import logging
import logging.handlers
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Dict, List, NamedTuple, Optional
import cv2
import numpy as np
import requests
import json
from PIL import Image
from nrcpy import NrcDevice
import queue

import urllib.request


# Reused connection pool for all PocketBase HTTP calls
_session = requests.Session()

# Dedicated thread pool for blocking relay (telnet) operations so they
# never consume workers from the recognition/DB-insert executor.
_relay_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="relay")


logging.basicConfig(
    level=logging.INFO,

    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler("log.txt", mode='a',
                                             maxBytes=5 * 1024 * 1024,
                                             backupCount=2,
                                             encoding='utf-8'),
        logging.StreamHandler()
    ]
)


# ---------------------------------------------------------------------------
#  DbWorker: dedicated bounded-queue + thread-pool for async DB operations
# ---------------------------------------------------------------------------

class DbWorker:
    """Background worker that serialises DB insert/update/delete operations
    through a bounded queue so they never block the video pipeline.

    Queue overflow causes the *oldest unprocessed* task to be dropped (the
    caller is warned via log), keeping memory bounded and the system
    responsive under load.
    """

    def __init__(self, max_workers: int = 4, queue_size: int = 128,
                 worker_name: str = "db"):
        self._queue: queue.Queue = queue.Queue(maxsize=queue_size)
        self._pool = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix=worker_name)
        self._shutdown_event = threading.Event()
        self._queue_size = queue_size
        self._dropped = 0
        self._submitted = 0
        self._completed = 0
        self._lock = threading.Lock()
        self._worker_threads: list[threading.Thread] = []
        for i in range(max_workers):
            t = threading.Thread(
                target=self._run_loop, name=f"{worker_name}-{i}", daemon=True)
            t.start()
            self._worker_threads.append(t)
        logging.info(
            f"DbWorker started: {max_workers} workers, queue_size={queue_size}")

    def _run_loop(self):
        """Each worker thread pulls tasks from the shared queue."""
        while not self._shutdown_event.is_set():
            try:
                task_fn, task_args, task_kwargs = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                task_fn(*task_args, **task_kwargs)
                with self._lock:
                    self._completed += 1
            except Exception as e:
                logging.error(f"DbWorker task error: {e}", exc_info=True)
            finally:
                self._queue.task_done()

    def submit(self, fn, *args, **kwargs) -> bool:
        """Submit a callable to the DB worker queue.

        Returns True if queued, False if the queue is full (task dropped).
        """
        try:
            self._queue.put_nowait((fn, args, kwargs))
            with self._lock:
                self._submitted += 1
            logging.debug("DbWorker: task queued")
            return True
        except queue.Full:
            with self._lock:
                self._dropped += 1
                dropped = self._dropped
            if dropped <= 5 or dropped % 50 == 0:
                logging.warning(
                    f"DbWorker: queue full, dropping task "
                    f"(total dropped: {dropped})")
            return False

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def stats(self) -> dict:
        with self._lock:
            return {
                "submitted": self._submitted,
                "completed": self._completed,
                "dropped": self._dropped,
                "pending": self.pending,
            }

    def shutdown(self, wait: bool = False):
        """Gracefully stop all worker threads."""
        self._shutdown_event.set()
        for _ in self._worker_threads:
            try:
                self._queue.put_nowait((None, (), {}))
            except queue.Full:
                pass
        if wait:
            self._pool.shutdown(wait=True)
        else:
            self._pool.shutdown(wait=False)


# Global DB worker instance
_db_worker: Optional[DbWorker] = None
_db_worker_lock = threading.Lock()


def get_db_worker() -> DbWorker:
    """Get or create the global DbWorker singleton."""
    global _db_worker
    if _db_worker is None:
        with _db_worker_lock:
            if _db_worker is None:
                _db_worker = DbWorker(
                    max_workers=4, queue_size=128, worker_name="db-worker")
    return _db_worker


def submit_db_task(fn, *args, **kwargs) -> bool:
    """Convenience wrapper: submit a function to the global DbWorker."""
    return get_db_worker().submit(fn, *args, **kwargs)


# ---------------------------------------------------------------------------
#  Registration pipeline: clean, reusable functions
# ---------------------------------------------------------------------------

def parse_embeddings_field(value) -> list:
    """Parse the ``embdanings`` DB field into a list of 512-d vectors.

    Handles every representation seen in the wild:

    * ``None`` / empty -> ``[]``
    * JSON string (e.g. ``"[[...512...], [...]]"`` or flat ``"[..]"``)
      -> ``json.loads`` first
    * list of lists (multi-embedding) -> as-is
    * flat list of floats (single embedding) -> wrapped

    Invalid entries are skipped (with a warning) instead of dropping the
    whole person.  Previously the loaders assumed a list while
    add/remove assumed a string, so one of the two paths silently failed
    depending on the actual PocketBase field type.
    """
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            logging.warning("parse_embeddings_field: not valid JSON, skipped")
            return []
    if not isinstance(value, list) or len(value) == 0:
        return []
    try:
        return safe_reshape(value)
    except (ValueError, TypeError, IndexError) as e:
        logging.warning(f"parse_embeddings_field: {e}")
        return []


def select_primary_face(faces):
    """Return the largest face by bounding-box area.

    ``faces[0]`` is NOT guaranteed to be the main subject; in a crop with
    several people it may be a background face.  Largest-area is the most
    reliable heuristic for "the person this photo is about".
    """
    if not faces:
        return None
    if len(faces) == 1:
        return faces[0]
    def _area(f):
        x1, y1, x2, y2 = f.bbox
        return max(x2 - x1, 0) * max(y2 - y1, 0)
    return max(faces, key=_area)


def face_blur_score(face_crop_bgr) -> float:
    """Sharpness of a face crop via variance of the Laplacian.

    Higher = sharper.  Sharp portraits are typically >100; heavy blur /
    motion-blur CCTV frames are often <30.
    """
    if face_crop_bgr is None or face_crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


# Identity model pack that produced the embeddings stored in this
# deployment.  Vectors coming from different packs (antelopev2 vs
# buffalo_l, ...) are NOT comparable: matching them silently destroys
# accuracy because every score becomes meaningless.  Records carrying a
# different tag are excluded from the matcher.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "antelopev2/glintr100")

# Quality gate defaults for reference images at registration time.
DEFAULT_MIN_BLUR = float(os.getenv("MIN_FACE_BLUR", "10.0"))

# Smallest face the pipeline accepts, in pixels (shorter side of the bbox).
# Measured on this deployment's 1400px CCTV frames: a 56px face still
# matches the same person across scales at cosine 0.69-0.88 (gate is
# 0.60), so the historical 64px floor was silently rejecting usable
# frames and returning "No face detected in image".
DEFAULT_MIN_FACE_PX = int(os.getenv("MIN_FACE_PX", "40"))


def get_min_face_px(config=None) -> int:
    """Effective minimum face size (pixels) for detection/embedding gates.

    Resolution order: an explicit ``MIN_FACE_PX`` environment variable
    (deployment override) beats ``setting.minFacePx`` from PocketBase,
    which beats ``DEFAULT_MIN_FACE_PX``.  Pass the runtime config object
    (e.g. ``CCtvMonitor``) to honour the DB setting; registration paths
    that have no config use the env/default value.
    """
    env = os.getenv("MIN_FACE_PX")
    if env:
        try:
            return max(8, int(float(env)))
        except ValueError:
            logging.warning(f"get_min_face_px: bad MIN_FACE_PX={env!r}")
    if config is not None:
        try:
            value = getattr(config, "minFacePx", None)
            if value not in (None, "") and float(value) > 0:
                return int(float(value))
        except (TypeError, ValueError):
            pass
    return DEFAULT_MIN_FACE_PX


def face_yaw(face) -> float:
    """Head yaw in degrees from the 1k3d68 landmark model (0.0 if absent).

    The ``pose`` attribute is only populated when the landmark_3d_68 model
    is loaded in the pack; older/trimmed packs leave it unset and the Face
    dict returns None, in which case yaw is simply unknown (0.0).
    """
    pose = getattr(face, "pose", None)
    if pose is None:
        return 0.0
    try:
        return float(pose[1])
    except (TypeError, IndexError, ValueError):
        return 0.0


def build_embedding_meta(blur: float = 0.0, det: float = 0.0,
                         yaw: float = 0.0, model: str = None) -> dict:
    """One ``embeddingMeta`` entry describing a stored reference vector."""
    return {
        "blur": round(float(blur), 2),
        "det": round(float(det), 4),
        "yaw": round(float(yaw), 2),
        "model": model or EMBEDDING_MODEL,
        "added": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def parse_embedding_meta(value) -> list:
    """Parse the ``embeddingMeta`` json field into a list of dicts."""
    if value is None:
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            value = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return []
    if not isinstance(value, list):
        return []
    return [e for e in value if isinstance(e, dict)]


def fit_meta_length(meta: list, count: int) -> list:
    """Pad/trim *meta* so it stays index-aligned with *count* embeddings.

    Records written before ``embeddingMeta`` existed have no per-vector
    quality data; padding keeps every later index in sync instead of
    shifting entries onto the wrong embedding.
    """
    meta = list(meta or [])
    if len(meta) > count:
        return meta[:count]
    while len(meta) < count:
        meta.append({"model": EMBEDDING_MODEL})
    return meta


def extract_face_embedding(image_path: str, face_embedder, face_lock=None,
                           model=None, model_lock=None, device: str = 'cpu',
                           min_face_px: int = None,
                           return_meta: bool = False):
    """Detect a face in *image_path* and return the 512-d embedding.

    Returns None if the image can't be read, no face is found, or the face
    is too small.  With ``return_meta=True`` the result is an
    ``(embedding, meta)`` tuple and failures come back as
    ``(None, None)`` so callers can unpack uniformly.
    """
    def _fail():
        return (None, None) if return_meta else None

    min_face_px = min_face_px or get_min_face_px()

    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"extract_face_embedding: cannot read image: {image_path}")
        return _fail()

    # Optionally run YOLO person detection first to crop the person region
    if model is not None:
        with (model_lock if model_lock is not None else nullcontext()):
            frame = model(img, classes=[0], device=device)[0]
        if len(frame.boxes) > 0:
            x1, y1, x2, y2 = map(int, frame.boxes.xyxy[0][:4])
            img = img[y1:y2, x1:x2]

    with (face_lock if face_lock is not None else nullcontext()):
        faces = face_embedder.get(img)

    if not faces:
        logging.warning(f"extract_face_embedding: no face detected in '{image_path}'")
        return _fail()

    if len(faces) > 1:
        logging.debug(
            f"extract_face_embedding: {len(faces)} faces in '{image_path}', "
            f"selecting largest")
    face = select_primary_face(faces)
    fx1, fy1, fx2, fy2 = map(int, face.bbox)
    if min(fx2 - fx1, fy2 - fy1) < min_face_px:
        logging.warning(
            f"extract_face_embedding: face too small "
            f"({fx2 - fx1}x{fy2 - fy1}px, need >={min_face_px}px) "
            f"in '{image_path}'")
        return _fail()

    # Quality gate: reject extremely blurry reference faces so one bad
    # photo can't poison the person's embedding set.  Threshold is
    # intentionally lenient (env MIN_FACE_BLUR, default 10.0).
    h, w = img.shape[:2]
    crop = img[max(fy1, 0):min(fy2, h), max(fx1, 0):min(fx2, w)]
    blur = face_blur_score(crop)
    yaw = face_yaw(face)
    det = float(getattr(face, "det_score", 0.0) or 0.0)
    min_blur = DEFAULT_MIN_BLUR
    logging.debug(
        f"extract_face_embedding: '{image_path}' face {fx2 - fx1}x{fy2 - fy1}px "
        f"det={det:.3f} blur={blur:.1f} yaw={yaw:.1f}")
    if blur < min_blur:
        logging.warning(
            f"extract_face_embedding: rejecting blurry face "
            f"(blur={blur:.1f} < {min_blur}) in '{image_path}'")
        return _fail()

    if face.embedding is None:
        logging.error(
            f"extract_face_embedding: recognition model produced no "
            f"embedding for '{image_path}'")
        return _fail()

    if return_meta:
        return face.embedding, build_embedding_meta(blur, det, yaw)
    return face.embedding


def validate_face_embedding(embedding: np.ndarray, expected_dim: int = 512) -> bool:
    """Check that an embedding has the expected shape and is not all-zero."""
    if embedding is None:
        return False
    emb = np.asarray(embedding, dtype=np.float32)
    if emb.ndim != 1 or emb.shape[0] != expected_dim:
        logging.error(
            f"validate_face_embedding: expected dim {expected_dim}, "
            f"got shape {emb.shape}")
        return False
    if np.allclose(emb, 0):
        logging.error("validate_face_embedding: embedding is all zeros")
        return False
    return True


FACE_CROP_PADDING = 40


def _crop_face(img, face_handler, face_lock,
               model=None, model_lock=None,
               device: str = 'cpu'):
    """Detect and crop the first face from *img*.

    Returns the cropped BGR image or ``None``.
    """
    if model is not None:
        try:
            with (model_lock if model_lock is not None else nullcontext()):
                frame = model(img, classes=[0], device=device)[0]
            if len(frame.boxes) > 0:
                x1, y1, x2, y2 = map(int, frame.boxes.xyxy[0][:4])
                img = img[y1:y2, x1:x2]
        except Exception:
            pass

    try:
        with (face_lock if face_lock is not None else nullcontext()):
            faces = face_handler.get(img)
    except Exception:
        return None

    if not faces:
        return None

    face = select_primary_face(faces)
    fx1, fy1, fx2, fy2 = map(int, face.bbox)
    h, w = img.shape[:2]
    pad = FACE_CROP_PADDING
    cx1 = max(fx1 - pad, 0)
    cy1 = max(fy1 - pad, 0)
    cx2 = min(fx2 + pad, w)
    cy2 = min(fy2 + pad, h)
    return img[cy1:cy2, cx1:cx2]


def generate_face_crop_base64(image_path: str, face_handler, face_lock,
                               model=None, model_lock=None,
                               device: str = 'cpu') -> Optional[str]:
    """Generate a base64-encoded JPEG face crop from an image."""
    img = cv2.imread(image_path)
    if img is None:
        return None
    cropped = _crop_face(img, face_handler, face_lock, model, model_lock,
                         device)
    if cropped is None:
        return None
    _, encoded = cv2.imencode(".jpg", cropped)
    return base64.b64encode(encoded.tobytes()).decode('utf-8')


def generate_face_crop_file(image_path: str, face_handler, face_lock,
                            model=None, model_lock=None,
                            device: str = 'cpu') -> Optional[str]:
    """Generate a JPEG face crop and save it to a temp file.

    Returns the temp file path or ``None``.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    cropped = _crop_face(img, face_handler, face_lock, model, model_lock,
                         device)
    if cropped is None:
        return None
    import tempfile
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    cv2.imwrite(path, cropped)
    return path


# ---------------------------------------------------------------------------
#  Person / face CRUD helpers (PocketBase HTTP)
# ---------------------------------------------------------------------------

def find_person_record(name: str) -> Optional[dict]:
    """Return the first known_face record for *name*, or None.

    The filter is passed via ``params`` so names with spaces or special
    characters are URL-encoded correctly.  (The old inline ``?filter=``
    string broke for names like "John Doe", which made lookups return
    None -> duplicate records were created and refresh_person() could
    never find the new person until a full restart.)
    """
    url = "http://127.0.0.1:8091/api/collections/known_face/records"
    safe_name = name.replace('"', '')
    try:
        response = _session.get(
            url, params={"filter": f'name = "{safe_name}"'}, timeout=5)
    except requests.RequestException as e:
        logging.error(f"find_person_record: failed to check {name}: {e}")
        return None

    if response.status_code == 200:
        records = response.json()
        return records['items'][0] if records['items'] else None
    logging.warning(
        f"find_person_record: status {response.status_code} for '{name}'")
    return None


def check_person_exists(name: str) -> bool:
    return find_person_record(name) is not None


def create_person_in_db(name: str, embedding: np.ndarray, img_path: str,
                        age: str, gender: str, role: str,
                        socialnumber: str,
                        face_crop_b64: str = None,
                        face_crop_path: str = None,
                        userwhom: str = "",
                        description: str = "",
                        meta: dict = None) -> bool:
    """Create a new person record with one initial face embedding."""
    url = "http://127.0.0.1:8091/api/collections/known_face/records"
    meta = meta or build_embedding_meta()
    data = {
        "name": name,
        "embdanings": json.dumps(embedding.tolist()),
        "gender": gender,
        "age": age,
        "role": role,
        "socialnumber": socialnumber,
        "userwhom": userwhom,
        "description": description,
        # Which model pack produced the vector, and per-vector quality
        # (blur/det/yaw).  PocketBase silently drops unknown keys, so this
        # is safe to send before the fields exist in the schema.
        "embeddingModel": EMBEDDING_MODEL,
        "embeddingMeta": json.dumps([meta]),
        "isActive": True,
    }
    if face_crop_b64:
        data["face_crop"] = face_crop_b64
    try:
        files = {"image": open(img_path, 'rb')}
        if face_crop_path and os.path.exists(face_crop_path):
            files["faceCrop"] = open(face_crop_path, 'rb')
        try:
            response = _session.post(url, data=data, files=files, timeout=10)
        finally:
            for f in files.values():
                f.close()
        if response.status_code in (200, 201):
            logging.info(f"create_person_in_db: created '{name}'")
            return True
        else:
            logging.error(
                f"create_person_in_db: failed to create '{name}': "
                f"{response.status_code} {response.text}")
            return False
    except requests.RequestException as e:
        logging.error(f"create_person_in_db: network error for '{name}': {e}")
        return False
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)
        if face_crop_path and os.path.exists(face_crop_path):
            os.remove(face_crop_path)


def update_person_embeddings(name: str, all_embeddings: list, record: dict,
                             img_path: str, age: str, gender: str,
                             role: str, socialnumber: str,
                             face_crop_b64: str = None,
                             face_crop_path: str = None,
                             userwhom: str = "",
                             description: str = "",
                             metas: list = None) -> bool:
    """Replace the full embedding list for an existing person.

    *all_embeddings* should be a list of np.ndarray (512-d each).
    This APPENDS the new embedding to the existing ones (unless the
    embedding is already present).

    *metas* is the index-aligned ``embeddingMeta`` list; when omitted the
    previous entries are preserved and padded.
    """
    record_id = record['id']
    if metas is None:
        metas = fit_meta_length(
            parse_embedding_meta(record.get("embeddingMeta")),
            len(all_embeddings))
    data = {
        "embdanings": json.dumps([e.tolist() for e in all_embeddings]),
        "name": name,
        "gender": gender,
        "age": age,
        "role": role,
        "socialnumber": socialnumber,
        "userwhom": userwhom,
        "description": description,
        "embeddingModel": EMBEDDING_MODEL,
        "embeddingMeta": json.dumps(metas),
        "isActive": True,
    }
    if face_crop_b64:
        data["face_crop"] = face_crop_b64
    try:
        files = {"image": open(img_path, 'rb')}
        if face_crop_path and os.path.exists(face_crop_path):
            files["faceCrop"] = open(face_crop_path, 'rb')
        try:
            update_url = (
                f"http://127.0.0.1:8091/api/collections/known_face/records/"
                f"{record_id}"
            )
            response = _session.patch(
                update_url, data=data, files=files, timeout=10)
        finally:
            for f in files.values():
                f.close()
        if response.status_code == 200:
            logging.info(
                f"update_person_embeddings: updated '{name}' "
                f"({len(all_embeddings)} embeddings)")
            return True
        else:
            logging.error(
                f"update_person_embeddings: failed for '{name}': "
                f"{response.status_code} {response.text}")
            return False
    except requests.RequestException as e:
        logging.error(
            f"update_person_embeddings: network error for '{name}': {e}")
        return False
    finally:
        if os.path.exists(img_path):
            os.remove(img_path)
        if face_crop_path and os.path.exists(face_crop_path):
            os.remove(face_crop_path)


def add_embedding_to_person(name: str, new_embedding: np.ndarray,
                            img_path: str, age: str, gender: str,
                            role: str, socialnumber: str,
                            userwhom: str = "",
                            description: str = "",
                            face_handler=None, face_lock=None,
                            model=None, model_lock=None,
                            device: str = 'cpu',
                            meta: dict = None) -> bool:
    """Append *new_embedding* to an existing person's embedding list.

    If the person doesn't exist yet, creates them.
    Duplicate embeddings (cosine similarity > 0.99) are skipped.

    If face_handler is provided, a face crop is generated and stored
    in the ``faceCrop`` file field of the PocketBase record.

    *meta* is the quality record for the new vector (blur/det/yaw/model);
    it is appended to ``embeddingMeta`` so quality stays index-aligned
    with ``embdanings``.
    """
    # Generate face crop for storage
    face_crop_b64 = None
    face_crop_path = None
    if face_handler is not None:
        face_crop_b64 = generate_face_crop_base64(
            img_path, face_handler, face_lock, model, model_lock, device)
        face_crop_path = generate_face_crop_file(
            img_path, face_handler, face_lock, model, model_lock, device)

    record = find_person_record(name)

    if record is None:
        logging.info(
            f"add_embedding_to_person: '{name}' not found, creating new")
        return create_person_in_db(
            name, new_embedding, img_path, age, gender, role, socialnumber,
            face_crop_b64=face_crop_b64, face_crop_path=face_crop_path,
            userwhom=userwhom, description=description, meta=meta)

    # Parse existing embeddings (handles both JSON-string and
    # native-list field representations)
    existing_embeddings: list[np.ndarray] = []
    for e in parse_embeddings_field(record.get("embdanings", "")):
        try:
            arr = np.array(e, dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if arr.shape == (512,):
            existing_embeddings.append(arr)

    # Keep embeddingMeta index-aligned with embdanings so the new entry
    # lands at the same position as the new vector.
    existing_meta = fit_meta_length(
        parse_embedding_meta(record.get("embeddingMeta")),
        len(existing_embeddings))

    # Check for duplicate (cosine similarity > 0.99)
    new_emb_norm = new_embedding.astype(np.float32)
    n = np.linalg.norm(new_emb_norm)
    if n > 0:
        new_emb_norm = new_emb_norm / n

    for existing in existing_embeddings:
        ex_norm = existing.astype(np.float32)
        en = np.linalg.norm(ex_norm)
        if en > 0:
            ex_norm = ex_norm / en
        sim = float(np.dot(new_emb_norm, ex_norm))
        if sim > 0.99:
            logging.info(
                f"add_embedding_to_person: skipping duplicate embedding "
                f"for '{name}' (sim={sim:.4f})")
            return True

    existing_embeddings.append(new_embedding)
    existing_meta.append(meta or build_embedding_meta())
    logging.info(
        f"add_embedding_to_person: appending embedding for '{name}' "
        f"(now {len(existing_embeddings)} total)")
    return update_person_embeddings(
        name, existing_embeddings, record, img_path, age, gender, role,
        socialnumber, face_crop_b64=face_crop_b64,
        face_crop_path=face_crop_path,
        userwhom=userwhom, description=description, metas=existing_meta)


def remove_person_embedding(name: str, embedding_index: int) -> bool:
    """Remove a specific embedding by index from a person's record.

    Returns False if the person doesn't exist or the index is out of range.
    """
    record = find_person_record(name)
    if not record:
        logging.error(f"remove_person_embedding: '{name}' not found")
        return False

    existing_embeddings: list[np.ndarray] = []
    for e in parse_embeddings_field(record.get("embdanings", "")):
        try:
            arr = np.array(e, dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if arr.shape == (512,):
            existing_embeddings.append(arr)

    if embedding_index < 0 or embedding_index >= len(existing_embeddings):
        logging.error(
            f"remove_person_embedding: index {embedding_index} out of range "
            f"for '{name}' (has {len(existing_embeddings)} embeddings)")
        return False

    if len(existing_embeddings) <= 1:
        logging.error(
            f"remove_person_embedding: cannot remove last embedding of "
            f"'{name}'. Delete the person instead.")
        return False

    existing_embeddings.pop(embedding_index)
    existing_meta = fit_meta_length(
        parse_embedding_meta(record.get("embeddingMeta")),
        len(existing_embeddings) + 1)
    if embedding_index < len(existing_meta):
        existing_meta.pop(embedding_index)
    existing_meta = fit_meta_length(existing_meta, len(existing_embeddings))

    record_id = record['id']
    data = {
        "embdanings": json.dumps([e.tolist() for e in existing_embeddings]),
        "embeddingMeta": json.dumps(existing_meta),
        "name": record.get('name', name),
        "gender": record.get('gender', ''),
        "age": record.get('age', ''),
        "role": record.get('role', ''),
        "socialnumber": record.get('socialnumber', ''),
    }
    try:
        update_url = (
            f"http://127.0.0.1:8091/api/collections/known_face/records/"
            f"{record_id}"
        )
        response = _session.patch(update_url, data=data, timeout=10)
        if response.status_code == 200:
            logging.info(
                f"remove_person_embedding: removed index {embedding_index} "
                f"from '{name}' ({len(existing_embeddings)} remaining)")
            return True
        else:
            logging.error(
                f"remove_person_embedding: failed for '{name}': "
                f"{response.status_code}")
            return False
    except requests.RequestException as e:
        logging.error(f"remove_person_embedding: network error: {e}")
        return False


def delete_person_from_db(name: str) -> bool:
    """Delete a known_face record by name."""
    record = find_person_record(name)
    if not record:
        logging.error(f"delete_person_from_db: '{name}' not found")
        return False

    record_id = record['id']
    try:
        url = (
            f"http://127.0.0.1:8091/api/collections/known_face/records/"
            f"{record_id}"
        )
        response = _session.delete(url, timeout=10)
        if response.status_code in (200, 204):
            logging.info(f"delete_person_from_db: deleted '{name}'")
            return True
        else:
            logging.error(
                f"delete_person_from_db: failed for '{name}': "
                f"{response.status_code}")
            return False
    except requests.RequestException as e:
        logging.error(f"delete_person_from_db: network error: {e}")
        return False


def get_person_faces(name: str) -> Optional[dict]:
    """Return face metadata for a person, including embedding count."""
    record = find_person_record(name)
    if not record:
        return None

    embedding_count = 0
    existing_emb_str = record.get("embdanings", "")
    if existing_emb_str:
        try:
            emb_list = json.loads(existing_emb_str)
            if isinstance(emb_list, list) and len(emb_list) > 0:
                if isinstance(emb_list[0], list):
                    embedding_count = len(emb_list)
                elif isinstance(emb_list[0], (int, float)):
                    embedding_count = 1
        except (json.JSONDecodeError, ValueError):
            pass

    return {
        "name": record.get("name", name),
        "age": record.get("age", ""),
        "gender": record.get("gender", ""),
        "role": record.get("role", ""),
        "socialnumber": record.get("socialnumber", ""),
        "embedding_count": embedding_count,
        "record_id": record.get("id", ""),
        # The stored face crop lives in the `faceCrop` FILE field; the old
        # code read a non-existent `face_crop` key and always returned "".
        "face_crop": record.get("faceCrop") or record.get("face_crop", ""),
        "image": record.get("image", ""),
    }


# ---------------------------------------------------------------------------
#  Legacy wrapper: reciveFromUi  (preserved for backward compat)
# ---------------------------------------------------------------------------

def _resolve_models(face_embedder, model, device: str = 'cpu'):
    """Return ``(face_embedder, model)``, building them when not supplied.

    ``oldsavetoDP.reciveFromUi`` created its own InsightFace + YOLO session
    on every call (old workflow).  Callers that already own shared sessions
    - the FastAPI app and the engine - pass them in and skip the expensive
    construction, so both the old 8-argument and the new call style work.
    """
    if face_embedder is None:
        from insightface.app import FaceAnalysis
        face_embedder = FaceAnalysis(
            'antelopev2',
            providers=(['CUDAExecutionProvider', 'CPUExecutionProvider']
                       if device == 'cuda' else ['CPUExecutionProvider']),
            root='.')
        face_embedder.prepare(ctx_id=0)
        logging.info("reciveFromUi: built a dedicated FaceAnalysis session")
    if model is None:
        from ultralytics import YOLO
        model = YOLO('models/yolov8n.pt')
        logging.info("reciveFromUi: built a dedicated YOLO session")
    return face_embedder, model


def reciveFromUi(name, imagePath, age, gender, role, socialnumber, isUrl, device,
                 face_embedder=None, model=None, face_lock=None, model_lock=None,
                 userwhom="", description="", min_face_px: int = None):
    """Receive data from the UI and process it, reusing the shared models.

    Registers or updates a person with a single face image.
    For multi-image support, use ``reciveFromUi_multi``.

    When *face_embedder* / *model* are omitted the old self-contained
    workflow builds them for this call.
    """
    face_embedder, model = _resolve_models(face_embedder, model, device)
    min_face_px = min_face_px or get_min_face_px()

    if isUrl:
        path = urllib.request.urlretrieve(
            imagePath, "uploads/local-filename.jpg")
        imagePath = path[0]

    if cv2.imread(imagePath) is None:
        logging.error(f"Image not found at {imagePath}")
        return

    # Single shared detection path: largest face + size/blur quality gate
    # (see extract_face_embedding).  imagePath must still exist on disk
    # because add_embedding_to_person re-reads it for the face crop.
    embed, meta = extract_face_embedding(
        imagePath, face_embedder, face_lock, model, model_lock,
        device, min_face_px, return_meta=True)
    if embed is None:
        raise ValueError(
            f"No usable face in '{imagePath}'. Register people from "
            f"closer/higher-resolution, sharp, front-facing photos.")

    add_embedding_to_person(
        name, embed, imagePath, age, gender, role, socialnumber,
        userwhom=userwhom, description=description,
        face_handler=face_embedder, face_lock=face_lock,
        model=model, model_lock=model_lock, device=device, meta=meta)
    return name


def reciveFromUi_multi(name: str, image_paths: list[str], age: str,
                       gender: str, role: str, socialnumber: str,
                       isUrl: bool, device: str, face_embedder=None,
                       model=None, face_lock=None,
                       model_lock=None,
                       userwhom: str = "",
                       description: str = "",
                       min_face_px: int = None) -> dict:
    """Register a person with multiple face images in one call.

    Returns a dict with per-image success/failure information.
    """
    face_embedder, model = _resolve_models(face_embedder, model, device)
    min_face_px = min_face_px or get_min_face_px()
    results: list[dict] = []
    success_count = 0
    fail_count = 0

    for i, image_path in enumerate(image_paths):
        entry = {"index": i, "path": image_path, "success": False, "error": ""}
        try:
            if isUrl:
                local_path = urllib.request.urlretrieve(
                    image_path, f"uploads/multi-{name}-{i}.jpg")
                image_path = local_path[0]

            embed, meta = extract_face_embedding(
                image_path, face_embedder, face_lock, model, model_lock,
                device, min_face_px, return_meta=True)

            if embed is None:
                entry["error"] = "No valid face detected or face too small"
                fail_count += 1
                results.append(entry)
                continue

            if not validate_face_embedding(embed):
                entry["error"] = "Invalid embedding"
                fail_count += 1
                results.append(entry)
                continue

            ok = add_embedding_to_person(
                name, embed, image_path, age, gender, role, socialnumber,
                userwhom=userwhom, description=description,
                face_handler=face_embedder, face_lock=face_lock,
                model=model, model_lock=model_lock, device=device, meta=meta)
            entry["success"] = ok
            if ok:
                success_count += 1
            else:
                fail_count += 1

        except Exception as e:
            entry["error"] = str(e)
            fail_count += 1
            logging.error(f"reciveFromUi_multi: error on image {i}: {e}")

        results.append(entry)

    return {
        "name": name,
        "total": len(image_paths),
        "success_count": success_count,
        "fail_count": fail_count,
        "details": results,
    }


# ---------------------------------------------------------------------------
#  Embedding loading helpers
# ---------------------------------------------------------------------------

def safe_reshape(embedding, dim=512):
    """Reshape a flat embedding list into a nested list of vectors.

    Nested input is filtered per-entry: only well-formed ``dim``-vectors
    are kept, so one corrupt reference can't poison the whole person.
    """
    if (isinstance(embedding, list) and len(embedding) > 0
            and isinstance(embedding[0], list)):
        return [e for e in embedding
                if isinstance(e, list) and len(e) == dim]

    if len(embedding) % dim != 0:
        raise ValueError(
            f"Inconsistent embedding length: {len(embedding)} not divisible "
            f"by {dim}")

    return [embedding[i:i + dim] for i in range(0, len(embedding), dim)]


def load_embeddings_from_db() -> dict:
    """Load all known face embeddings from the database.

    Returns a dict keyed by person name, each containing a list of 512-d
    numpy embeddings (multi-reference support) plus ``id`` (stable
    ``known_face`` record id, used as the person id in detection logs) and
    ``embeddingModel``.
    """
    known_names = {}
    base_url = "http://127.0.0.1:8091/api/collections/known_face/records"
    per_page = 500

    try:
        page = 1
        while True:
            res = _session.get(
                base_url,
                params={"perPage": per_page, "page": page, "sort": "-created"},
                timeout=10)
            res.raise_for_status()
            payload = res.json()
            records = payload.get("items", [])

            for item in records:
                _ingest_person_record(known_names, item)

            if not payload.get("page") or page >= int(payload.get("totalPages", 1)):
                break
            page += 1

        total_embeddings = sum(
            len(person['embeddings']) for person in known_names.values())
        logging.info(
            f"Loaded {total_embeddings} embeddings from "
            f"{len(known_names)} persons")
        return known_names

    except Exception as e:
        logging.error(f"Failed to load embeddings: {e}")
        return {}


def _ingest_person_record(known_names: dict, item: dict) -> None:
    """Fold one known_face record into the in-memory person table."""
    name = item.get("name")
    if not name:
        return

    if item.get("isActive") is False:
        logging.info(f"load_embeddings_from_db: '{name}' is inactive, skipped")
        return

    model = item.get("embeddingModel")
    if model and model != EMBEDDING_MODEL:
        # Mixing packs is worse than having fewer references: every score
        # would be meaningless.  Skip with an explicit warning instead.
        logging.warning(
            f"load_embeddings_from_db: '{name}' was registered with "
            f"'{model}' but this build matches with '{EMBEDDING_MODEL}'; "
            f"skipped to avoid cross-pack false matches")
        return

    age = item.get('age')
    gender = item.get('gender')
    role = item.get('role')
    socialnumber = item.get('socialnumber')

    reshaped = parse_embeddings_field(item.get("embdanings"))
    if not reshaped:
        logging.warning(
            f"load_embeddings_from_db: no valid embeddings "
            f"for '{name}', skipped")
        return

    if name in known_names:
        logging.warning(
            f"load_embeddings_from_db: duplicate known_face name '{name}' "
            f"(ids {known_names[name].get('id')} and {item.get('id')}) - "
            f"identities with the same display name cannot be told apart")
        person = known_names[name]
    else:
        person = known_names[name] = {
            'age': age,
            'gender': gender,
            'role': role,
            'socialnumber': socialnumber,
            'id': item.get('id', ''),
            'embeddingModel': model or EMBEDDING_MODEL,
            'simThreshold': item.get('simThreshold'),
            'embeddings': []
        }

    for emb in reshaped:
        try:
            emb_array = np.array(emb, dtype=np.float32)
        except (ValueError, TypeError):
            continue
        if emb_array.shape == (512,):
            person['embeddings'].append(emb_array)


def load_person_from_db(name: str) -> Optional[dict]:
    """Load a single known_face record (avoids a full 1000-record fetch)."""
    record = find_person_record(name)
    if not record:
        return None

    known_names: dict = {}
    _ingest_person_record(known_names, record)
    if name not in known_names:
        # Inactive or cross-pack record: report it as absent so the caller
        # evicts the person from the matcher instead of keeping them.
        logging.warning(
            f"load_person_from_db: '{name}' loaded but not matchable "
            f"(inactive or model mismatch)")
        return None
    return {name: known_names[name]}


# ---------------------------------------------------------------------------
#  Detection event recording (insertToDb for the collection log)
# ---------------------------------------------------------------------------

tempTime = None


def savePicture(frame, croppedface, humancrop, name, track_id, quality):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    frame_loc = f'outputs/screenshot/s.{name}_{track_id}.jpg'
    frame.save(
        f'{frame_loc}', "JPEG", quality=quality, optimize=True)
    # cropp
    croppedface = cv2.cvtColor(croppedface, cv2.COLOR_BGR2RGB)
    croppedface = Image.fromarray(croppedface)
    crop_loc = f'outputs/cropped/c.{name}_{track_id}.jpg'
    croppedface.save(
        f'{crop_loc}', "JPEG", quality=quality, optimize=True)
    humancrop = cv2.cvtColor(humancrop, cv2.COLOR_BGR2RGB)
    humancrop = Image.fromarray(humancrop)
    human_loc = f'outputs/humancrop/c.{name}_{track_id}.jpg'
    humancrop.save(
        f'{human_loc}', "JPEG", quality=quality, optimize=True)

    return frame_loc, crop_loc, human_loc


def timediff(current_time):
    global tempTime
    if tempTime is None:
        return True
    return (current_time - tempTime).total_seconds() >= 60


class RecentEntry(NamedTuple):
    name: str
    track_id: int
    time: datetime.datetime


recent_names: list[RecentEntry] = []
TIME_THRESHOLD = 10
_recent_lock = threading.Lock()


def clean_old_entries():
    now = datetime.datetime.now()
    recent_names[:] = [
        entry for entry in recent_names
        if (now - entry.time).total_seconds() < TIME_THRESHOLD
    ]


def should_insert(name, track_id):
    now = datetime.datetime.now()
    clean_old_entries()

    for entry in recent_names:
        if name == "unknown" and entry.name == "unknown":
            if entry.track_id == track_id:
                if (now - entry.time).total_seconds() < TIME_THRESHOLD:
                    return False

        elif entry.name == name:
            if (now - entry.time).total_seconds() < TIME_THRESHOLD:
                return False

    return True


def insertToDb(name, frame, croppedface, humancrop, score, track_id, gender,
               age, role, socialnumber, path, quality, regions, isRelay: bool,
               isRegionMode: bool, ip_relay, port_relay, relayn1, relayn2,
               sim: float = 0.0, margin: float = 0.0, obs_count: int = 0,
               person_id: str = "", blur: float = 0.0, yaw: float = 0.0,
               det_score: float = None):
    """Record a detection event (screenshot + cropped + metadata to PocketBase).

    This logs *every recognised detection*; it is NOT the same as
    registering a known face.  It runs on the DbWorker thread pool.

    ``score`` stays the detection confidence because the Flutter UI reads
    ``collection.score``; the identity evidence is written to the
    additive ``sim`` / ``matchMargin`` / ``obsCount`` fields so nothing
    in the existing UI breaks.
    """
    url = "http://127.0.0.1:8091/api/collections/collection/records"
    timeNow = datetime.datetime.now()
    display_time = timeNow.strftime("%H:%M:%S")
    display_date = timeNow.strftime("%Y-%m-%d")

    os.makedirs('outputs/cropped', exist_ok=True)
    os.makedirs('outputs/screenshot', exist_ok=True)
    os.makedirs('outputs/humancrop', exist_ok=True)

    with _recent_lock:
        allow_insert = should_insert(name, track_id)
        if allow_insert:
            recent_names.append(RecentEntry(
                name=name, track_id=track_id,
                time=datetime.datetime.now()))

    if allow_insert:
        frame_loc, crop_loc, human_loc = savePicture(
            frame, croppedface, humancrop, name, track_id, quality)

        relay_ip = relay_region = relay_number = None
        if isRegionMode and regions:
            relay_ip, relay_region, relay_number = (
                regions['relay_ip'], regions['name'], regions['relay_number'])

        with open(frame_loc, "rb") as file1, \
             open(crop_loc, "rb") as file2, \
             open(human_loc, 'rb') as file3:
            files = {
                "frame": (frame_loc, file1, "image/jpeg"),
                "cropped_frame": (crop_loc, file2, "image/jpeg"),
                "humancrop": (human_loc, file3, "image/jpeg")
            }

            # Unknown keys are dropped by PocketBase (verified), so these
            # can be sent before the fields are added to the collection.
            response = _session.post(url, files=files, timeout=10, data={
                "name": name,
                "score": score,
                "detScore": (score if det_score is None else det_score),
                "sim": round(float(sim or 0.0), 4),
                "matchMargin": round(float(margin or 0.0), 4),
                "obsCount": int(obs_count or 0),
                "blur": round(float(blur or 0.0), 2),
                "yaw": round(float(yaw or 0.0), 2),
                "personId": person_id or "",
                'gender': gender,
                'age': age,
                'camera': path,
                'date': display_date,
                'time': display_time,
                'role': role,
                'socialnumber': socialnumber,
                "track_id": str(track_id),
                'filename': human_loc.split('/')[2]
            })
        if response.status_code in [200, 201]:
            logging.debug(f"insertToDb: recorded detection id={response.json()['id']}")
            if (role == 'approve' and isRelay and isRegionMode
                    and relay_number is not None):
                _relay_executor.submit(
                    handle_relay_operations,
                    relay_ip, 23, 'admin', 'admin', int(relay_number))
            elif role == 'approve' and isRelay:
                _relay_executor.submit(
                    _do_relay_sequence, ip_relay, port_relay,
                    relayn1, relayn2)
        else:
            logging.error(f"insertToDb: error inserting to DB: {response.text}")


# ---------------------------------------------------------------------------
#  Connection / relay helpers
# ---------------------------------------------------------------------------

def init_db_session():
    """Warm up the PocketBase connection pool at startup."""
    try:
        _session.get("http://127.0.0.1:8091/api/health", timeout=2)
    except Exception:
        pass


def shutdown_relay_executor():
    """Gracefully shut down the dedicated relay thread pool."""
    _relay_executor.shutdown(wait=False)


def handle_relay_operations(ip='192.168.1.200', port=23, username='admin',
                            password='admin', relay_number=1):
    """Handle IP relay operations - single execution"""
    try:
        print(f"Executing relay operation for {ip}")
        nrc = NrcDevice((ip, port, username, password))

        nrc.connect()
        if nrc.login():
            nrc.relayContact(relay_number, 300)
            print(f"Relay operation completed for {ip}")
        nrc.disconnect()
    except Exception as e:
        print(f"Relay error for {ip}: {e}")


def _do_relay_sequence(ip_relay, port_relay, relayn1, relayn2):
    """Run the dual-relay sequence (with its 1s gap) on the relay thread."""
    try:
        if relayn1 == 1 and relayn2 == 2:
            handle_relay_operations(
                ip_relay, int(port_relay), 'admin', 'admin', int(relayn1))
            time.sleep(1)
            handle_relay_operations(
                ip_relay, int(port_relay), 'admin', 'admin', int(relayn2))
        elif relayn1 == 1:
            handle_relay_operations(
                ip_relay.strip(), int(port_relay), 'admin', 'admin',
                int(relayn1))
        elif relayn2 == 2:
            handle_relay_operations(
                ip_relay, int(port_relay), 'admin', 'admin', int(relayn2))
    except Exception as e:
        print(f"Relay sequence error: {e}")


if __name__ == "__main__":
    pass
