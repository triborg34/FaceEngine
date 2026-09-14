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

def extract_face_embedding(image_path: str, face_embedder, face_lock=None,
                           model=None, model_lock=None, device: str = 'cpu',
                           min_face_px: int = 64) -> Optional[np.ndarray]:
    """Detect a face in *image_path* and return the 512-d embedding.

    Returns None if the image can't be read, no face is found, or the face
    is too small.
    """
    img = cv2.imread(image_path)
    if img is None:
        logging.error(f"extract_face_embedding: cannot read image: {image_path}")
        return None

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
        return None

    face = faces[0]
    fx1, fy1, fx2, fy2 = map(int, face.bbox)
    if min(fx2 - fx1, fy2 - fy1) < min_face_px:
        logging.warning(
            f"extract_face_embedding: face too small "
            f"({fx2 - fx1}x{fy2 - fy1}px, need >={min_face_px}px) "
            f"in '{image_path}'")
        return None

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

    face = faces[0]
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
    """Return the first known_face record for *name*, or None."""
    url = (
        f"http://127.0.0.1:8091/api/collections/known_face/records"
        f"?filter=name=%22{name}%22"
    )
    try:
        response = _session.get(url, timeout=5)
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
                        description: str = "") -> bool:
    """Create a new person record with one initial face embedding."""
    url = "http://127.0.0.1:8091/api/collections/known_face/records"
    data = {
        "name": name,
        "embdanings": json.dumps(embedding.tolist()),
        "gender": gender,
        "age": age,
        "role": role,
        "socialnumber": socialnumber,
        "userwhom": userwhom,
        "description": description,
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
                             description: str = "") -> bool:
    """Replace the full embedding list for an existing person.

    *all_embeddings* should be a list of np.ndarray (512-d each).
    This APPENDS the new embedding to the existing ones (unless the
    embedding is already present).
    """
    record_id = record['id']
    data = {
        "embdanings": json.dumps([e.tolist() for e in all_embeddings]),
        "name": name,
        "gender": gender,
        "age": age,
        "role": role,
        "socialnumber": socialnumber,
        "userwhom": userwhom,
        "description": description,
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
                            device: str = 'cpu') -> bool:
    """Append *new_embedding* to an existing person's embedding list.

    If the person doesn't exist yet, creates them.
    Duplicate embeddings (cosine similarity > 0.99) are skipped.

    If face_handler is provided, a face crop is generated and stored
    in the ``face_crop`` field of the PocketBase record.
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
            userwhom=userwhom, description=description)

    # Parse existing embeddings
    existing_emb_str = record.get("embdanings", "")
    existing_embeddings: list[np.ndarray] = []
    if existing_emb_str:
        try:
            emb_list = json.loads(existing_emb_str)
            if isinstance(emb_list, list) and len(emb_list) > 0:
                if isinstance(emb_list[0], list):
                    # Multiple embeddings stored as list of lists
                    for e in emb_list:
                        arr = np.array(e, dtype=np.float32)
                        if arr.shape[0] == 512:
                            existing_embeddings.append(arr)
                elif isinstance(emb_list[0], (int, float)):
                    # Single embedding stored as flat list
                    arr = np.array(emb_list, dtype=np.float32)
                    if arr.shape[0] == 512:
                        existing_embeddings.append(arr)
        except (json.JSONDecodeError, ValueError) as e:
            logging.error(
                f"add_embedding_to_person: failed to parse embeddings "
                f"for '{name}': {e}")

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
    logging.info(
        f"add_embedding_to_person: appending embedding for '{name}' "
        f"(now {len(existing_embeddings)} total)")
    return update_person_embeddings(
        name, existing_embeddings, record, img_path, age, gender, role,
        socialnumber, face_crop_b64=face_crop_b64,
        face_crop_path=face_crop_path,
        userwhom=userwhom, description=description)


def remove_person_embedding(name: str, embedding_index: int) -> bool:
    """Remove a specific embedding by index from a person's record.

    Returns False if the person doesn't exist or the index is out of range.
    """
    record = find_person_record(name)
    if not record:
        logging.error(f"remove_person_embedding: '{name}' not found")
        return False

    existing_emb_str = record.get("embdanings", "")
    existing_embeddings: list[np.ndarray] = []
    if existing_emb_str:
        try:
            emb_list = json.loads(existing_emb_str)
            if isinstance(emb_list, list) and len(emb_list) > 0:
                if isinstance(emb_list[0], list):
                    for e in emb_list:
                        arr = np.array(e, dtype=np.float32)
                        if arr.shape[0] == 512:
                            existing_embeddings.append(arr)
                elif isinstance(emb_list[0], (int, float)):
                    arr = np.array(emb_list, dtype=np.float32)
                    if arr.shape[0] == 512:
                        existing_embeddings.append(arr)
        except (json.JSONDecodeError, ValueError):
            pass

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

    record_id = record['id']
    data = {
        "embdanings": json.dumps([e.tolist() for e in existing_embeddings]),
        "name": name,
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
        "face_crop": record.get("face_crop", ""),
        "image": record.get("image", ""),
    }


# ---------------------------------------------------------------------------
#  Legacy wrapper: reciveFromUi  (preserved for backward compat)
# ---------------------------------------------------------------------------

def reciveFromUi(name, imagePath, age, gender, role, socialnumber, isUrl, device,
                 face_embedder, model, face_lock=None, model_lock=None,
                 userwhom="", description=""):
    """Receive data from the UI and process it, reusing the shared models.

    Registers or updates a person with a single face image.
    For multi-image support, use ``reciveFromUi_multi``.
    """
    min_face_px = int(os.getenv("MIN_FACE_PX", "64"))

    if isUrl:
        path = urllib.request.urlretrieve(
            imagePath, "uploads/local-filename.jpg")
        imagePath = path[0]

    img = cv2.imread(imagePath)
    if img is None:
        logging.error(f"Image not found at {imagePath}")
        return

    if model is not None:
        with (model_lock if model_lock is not None else nullcontext()):
            frame = model(img, classes=[0], device=device)[0]
        if len(frame.boxes) > 0:
            x1, y1, x2, y2 = map(int, frame.boxes.xyxy[0][:4])
            img = img[y1:y2, x1:x2]

    with (face_lock if face_lock is not None else nullcontext()):
        face = face_embedder.get(img)

    if face:
        fx1, fy1, fx2, fy2 = map(int, face[0].bbox)
        if min(fx2 - fx1, fy2 - fy1) < min_face_px:
            raise ValueError(
                f"Face too small ({fx2 - fx1}x{fy2 - fy1}px, "
                f"need >={min_face_px}px) in '{imagePath}'. "
                f"Register people from closer/higher-resolution photos.")

        embed = face[0].embedding
        add_embedding_to_person(
            name, embed, imagePath, age, gender, role, socialnumber,
            userwhom=userwhom, description=description,
            face_handler=face_embedder, face_lock=face_lock,
            model=model, model_lock=model_lock, device=device)
        return name


def reciveFromUi_multi(name: str, image_paths: list[str], age: str,
                       gender: str, role: str, socialnumber: str,
                       isUrl: bool, device: str, face_embedder,
                       model=None, face_lock=None,
                       model_lock=None,
                       userwhom: str = "",
                       description: str = "") -> dict:
    """Register a person with multiple face images in one call.

    Returns a dict with per-image success/failure information.
    """
    min_face_px = int(os.getenv("MIN_FACE_PX", "64"))
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

            embed = extract_face_embedding(
                image_path, face_embedder, face_lock, model, model_lock,
                device, min_face_px)

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
                model=model, model_lock=model_lock, device=device)
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
    """Reshape a flat embedding list into a nested list of vectors."""
    if isinstance(embedding[0], list) and len(embedding[0]) == dim:
        return embedding

    if len(embedding) % dim != 0:
        raise ValueError(
            f"Inconsistent embedding length: {len(embedding)} not divisible "
            f"by {dim}")

    return [embedding[i:i + dim] for i in range(0, len(embedding), dim)]


def load_embeddings_from_db() -> dict:
    """Load all known face embeddings from the database.

    Returns a dict keyed by person name, each containing a list of 512-d
    numpy embeddings (multi-reference support).
    """
    known_names = {}
    url = "http://127.0.0.1:8091/api/collections/known_face/records?perPage=1000"

    try:
        res = _session.get(url, timeout=5)
        res.raise_for_status()
        records = res.json()["items"]

        for item in records:
            name = item["name"]
            embedding = item.get("embdanings")
            age = item.get('age')
            gender = item.get('gender')
            role = item.get('role')
            socialnumber = item.get('socialnumber')

            if embedding:
                embedding = embedding[:len(embedding) - (len(embedding) % 512)]
                try:
                    reshaped = safe_reshape(embedding)

                    if name not in known_names:
                        known_names[name] = {
                            'age': age,
                            'gender': gender,
                            'role': role,
                            'socialnumber': socialnumber,
                            'embeddings': []
                        }

                    for emb in reshaped:
                        emb_array = np.array(emb, dtype=np.float32)
                        known_names[name]['embeddings'].append(emb_array)

                except Exception as reshape_error:
                    logging.error(
                        f"Error reshaping embedding for {name}: "
                        f"{reshape_error}")

        total_embeddings = sum(
            len(person['embeddings']) for person in known_names.values())
        logging.info(
            f"Loaded {total_embeddings} embeddings from "
            f"{len(known_names)} persons")
        return known_names

    except Exception as e:
        logging.error(f"Failed to load embeddings: {e}")
        return {}


def load_person_from_db(name: str) -> Optional[dict]:
    """Load a single known_face record (avoids a full 1000-record fetch)."""
    record = find_person_record(name)
    if not record:
        return None

    age = record.get('age')
    gender = record.get('gender')
    role = record.get('role')
    socialnumber = record.get('socialnumber')
    embedding = record.get("embdanings")

    if not embedding:
        return {name: {
            'age': age, 'gender': gender, 'role': role,
            'socialnumber': socialnumber, 'embeddings': []
        }}

    embedding = embedding[:len(embedding) - (len(embedding) % 512)]
    try:
        reshaped = safe_reshape(embedding)
    except Exception as reshape_error:
        logging.error(f"Error reshaping embedding for {name}: {reshape_error}")
        return None

    embeddings = [np.array(emb, dtype=np.float32) for emb in reshaped]
    return {name: {
        'age': age, 'gender': gender, 'role': role,
        'socialnumber': socialnumber, 'embeddings': embeddings
    }}


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
               isRegionMode: bool, ip_relay, port_relay, relayn1, relayn2):
    """Record a detection event (screenshot + cropped + metadata to PocketBase).

    This logs *every recognised detection*; it is NOT the same as
    registering a known face.  It runs on the DbWorker thread pool.
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

            response = _session.post(url, files=files, timeout=10, data={
                "name": name,
                "score": score,
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
