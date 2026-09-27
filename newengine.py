
import asyncio
import datetime
import gc
import logging
import logging.handlers
import multiprocessing
import os
import platform
import queue
import subprocess
import time
import threading
import webbrowser
import requests
from torchvision.models import resnet50
from urllib.parse import urlparse
import cv2
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
import torch
from concurrent.futures import ThreadPoolExecutor
from camera import FreshestFrame
from newsavatoDb import (
    load_embeddings_from_db, load_person_from_db, insertToDb,
    get_db_worker, submit_db_task, select_primary_face,
    face_blur_score, face_yaw, get_min_face_px, DEFAULT_MIN_FACE_PX,
)
from PIL import Image
from torchvision.transforms import transforms
import json


# --- Basic Setup ---
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.handlers.RotatingFileHandler("log.txt", mode='a',
                                             maxBytes=5 * 1024 * 1024,
                                             backupCount=2, encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Leave headroom for the per-camera pipeline threads instead of letting
# every OpenCV call try to occupy all cores.
cv2.setNumThreads(max(1, min(4, multiprocessing.cpu_count())))

# --- Constants ---
FACE_CROP_PADDING = 40
SIMILARITY_THRESHOLD = 0.5
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.5
# SCRFD detection operating point.  Defaults preserve the previous
# behaviour (InsightFace default det_thresh=0.5, det_size=640).  Lower
# FACE_DET_THRESH (e.g. 0.35-0.4) recovers small/blurry/side-profile faces
# at the cost of more false candidates; raise FACE_DET_SIZE (e.g. 800-960)
# to keep small faces resolvable in high-res frames at the cost of speed.
# Tune with: python benchmark.py --threads N
FACE_DET_THRESH = float(os.getenv("FACE_DET_THRESH", "0.5"))
FACE_DET_SIZE = int(os.getenv("FACE_DET_SIZE", "640"))


def prepare_face_handler(handler):
    """Apply the configured detection operating point to a FaceAnalysis.

    Centralises what used to be bare ``prepare(ctx_id=0)`` calls so every
    session (shared, per-camera, UI/crop) runs the same detector config.
    """
    handler.prepare(
        ctx_id=0, det_thresh=FACE_DET_THRESH,
        det_size=(FACE_DET_SIZE, FACE_DET_SIZE))
    logging.info(
        f"FaceAnalysis prepared: det_thresh={FACE_DET_THRESH} "
        f"det_size={FACE_DET_SIZE}")
    return handler


RECOGNITION_UPDATE_INTERVAL = 2  # seconds
# Minimum gap between recognition attempts on the SAME track.  The old
# pipeline recognised a track exactly once for its whole lifetime, so a
# single unlucky frame (blur, profile, partial occlusion) decided the
# identity forever.  A short gap lets us collect several observations and
# vote on them instead.
OBS_INTERVAL = float(os.getenv("OBS_INTERVAL", "0.5"))
# Fallbacks for the new `setting` fields.  They are read from PocketBase
# at startup and ignored by PocketBase until the fields are added, so the
# defaults keep behaviour defined in one place.
DEFAULT_MARGIN_MIN = 0.06      # best-vs-2nd cosine gap for a trusted match
DEFAULT_VOTE_OBS = 5           # observations needed before committing
DEFAULT_VOTE_MAJORITY = 0.6    # fraction of observations that must agree
DEFAULT_MIN_BLUR = 10.0        # Laplacian variance gate for observations
DEFAULT_MAX_YAW = 35.0         # |yaw| gate in degrees
STALE_TRACK_TTL = 30  # seconds before state for unseen track IDs is pruned
JPEG_QUALITY = 85
# Smallest face this pipeline accepts (shorter bbox side, px).  The old
# 64px floor was measured against a wrong assumption: same identity across
# scales still scores 0.69-0.88 at 25-56px, so it only threw away usable
# frames ("no face detected").  Resolution (env MIN_FACE_PX >
# setting.minFacePx > default) lives in savatoDb.get_min_face_px.
MIN_FACE_PX = get_min_face_px()

# Recognition gates stored in the `setting` collection.  A stored 0 means
# "not configured" and never "no gate": PocketBase backfills the number
# columns it just added with 0 on pre-existing rows, and 0 is not a
# meaningful value for any of them - maxYaw=0 rejects every face (|yaw|>0
# is always true), minBlur=0 disables the sharpness check, voteObs=0 makes
# the fusion need only 2 frames.  Values are resolved here so the zeroed
# defaults cannot silently disable the whole accuracy layer.
_SETTING_GATES = (
    ('marginMin', DEFAULT_MARGIN_MIN, float),
    ('voteObs', DEFAULT_VOTE_OBS, int),
    ('voteMajority', DEFAULT_VOTE_MAJORITY, float),
    ('minBlur', DEFAULT_MIN_BLUR, float),
    ('maxYaw', DEFAULT_MAX_YAW, float),
    ('minFacePx', DEFAULT_MIN_FACE_PX, int),
)


def resolve_setting_gates(data) -> dict:
    """Gate values from a ``setting`` record; 0 / absent / junk -> default."""
    data = data or {}
    gates = {}
    for key, default, cast in _SETTING_GATES:
        try:
            value = data.get(key)
            number = float(default) if value in (None, "") else float(value)
        except (TypeError, ValueError):
            number = float(default)
        gates[key] = cast(number) if number > 0 else cast(default)
    return gates
# Dedicated InsightFace sessions per camera (opt-in via env). Measured on
# this machine: one SHARED session sustains ~88 calls/s across camera
# threads while concurrent sessions collapse to ~9 calls/s (GPU context
# thrashing), so the default keeps everything on the shared session.
# Run `python benchmark.py --threads N` before enabling on other GPUs.
MAX_FACE_SESSIONS = int(os.getenv("MAX_FACE_SESSIONS", "0"))
PERF_LOG_INTERVAL = float(os.getenv("PERF_LOG_INTERVAL", "30"))  # seconds


class _StageStats:
    """Thread-safe rolling stage timings, flushed to the log periodically.

    Usage: PERF_STATS.add("cam0:yolo", elapsed_seconds)
    """

    def __init__(self, log_interval=PERF_LOG_INTERVAL):
        self.log_interval = log_interval
        self._lock = threading.Lock()
        self._sums = {}
        self._counts = {}
        self._last_log = time.time()

    def add(self, stage, seconds):
        with self._lock:
            self._sums[stage] = self._sums.get(stage, 0.0) + seconds
            self._counts[stage] = self._counts.get(stage, 0) + 1
            now = time.time()
            if now - self._last_log >= self.log_interval:
                self._flush(now)

    def _flush(self, now):
        parts = []
        for stage in sorted(self._sums):
            count = max(self._counts[stage], 1)
            total = self._sums[stage]
            parts.append(f"{stage}: {1000*total/count:.1f}ms avg x{count}")
        if parts:
            logging.info("[perf] " + "; ".join(parts))
        self._sums.clear()
        self._counts.clear()
        self._last_log = now


PERF_STATS = _StageStats()

class CCtvMonitor:
    def __init__(self,device):
        self.process = None
        self.start()
        self.device = device
        self.frps = 5 if self.device == 'cuda' else 25
        self.fileEx = 'onnx' if self.checkOnnx() else 'pt'
        self.MODEL_PATH = os.getenv(
            "MODEL_PATH", f"models/yolov8n.{self.fileEx}")
        self.TARGET_FPS = 30
        self.FRAME_DELAY = 1.0 / self.TARGET_FPS
        self.RETRY_LIMIT = 5
        self.RETRY_DELAY = 3
        self.ip_relay, self.ip_port, self.relayN1, self.relayN2 = '', '', '', ''
        # loadConfig returns every setting as a dict (including the new
        # recognition gates); spread it onto the instance in one shot.
        self.__dict__.update(self.loadConfig())

        # Initialize models
        self.model = None
        self.face_handler = None
        # Dedicated per-camera InsightFace sessions bookkeeping
        self._face_sessions_created = 0
        self._face_sessions_disabled = False
        # The shared InsightFace/YOLO sessions get serialized across threads
        self.face_lock = threading.Lock()
        self.model_lock = threading.Lock()
        self._load_models()
        self.known_names = self.load_db()

        # Thread-safe embedding index: atomic swap via tuple assignment.
        # The index lock serialises rebuilds while readers use the old
        # snapshot until the new one is published.
        self._index_lock = threading.Lock()
        self._build_embedding_index()

        # Threading and process management
        self.embedding_cache = {}
        self._cache_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._shutdown_event = threading.Event()

        # Image Searcher
        self.FOLDER_PATH = "outputs/humancrop"             # folder containing all images
        self.EMBEDDING_FILE = "embeddings.npy"  # file to save/load embeddings
        self.FILENAMES_FILE = "filenames.txt"  # file to save/load filenames
        self.LOCAL_WEIGHTS = "models/resnet50-0676ba61.pth"
        self.IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
        self._image_searcher_model = None
        self._image_searcher_lock = threading.Lock()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # regions


        # self.loadWebBrowser(self.port)

    def checkOnnx(self):
        directory = 'models'
        for filename in os.listdir(directory):
            # Get full file path
            filepath = os.path.join(directory, filename)

            # Check if it's a file (not a directory)
            if os.path.isfile(filepath):
                # Check if filename is exactly "onnx"
                if filename == "onnx":
                    logging.info(f"Found the 'onnx' file!")

                    # Read and process the onnx file
                    return True
        return False

    def checkOpenVino(self):
        directory = 'models'
        for filename in os.listdir(directory):
            # Get full file path
            filepath = os.path.join(directory, filename)

            # Check if it's a file (not a directory)
            if os.path.isfile(filepath):
                # Check if filename is exactly "openvivo"
                if filename == "openvino":
                    logging.info(f"Found the 'openvivo' file!")

                    # Read and process the onnx file
                    return True
        return False

    def loadWebBrowser(self, port):
        webbrowser.open(f'http://127.0.0.1:{port}/web/app')

    def loadConfig(self):
        """Read runtime configuration from PocketBase.

        Returns a dict of every config attribute.  Retries for up to ~30s
        because ``__init__`` starts PocketBase and used to read it in the
        same tick: whenever the DB was slow the connection was refused and
        startup died with an unhandled exception.
        """
        try:
            with open('iou.txt') as file:
                iou = float(file.readline().strip())
        except (OSError, ValueError) as e:
            logging.warning(f"loadConfig: bad iou.txt ({e}), using 0.45")
            iou = 0.45

        uri = 'http://127.0.0.1:8091/api/collections/setting/records'
        data = None
        last_err = None
        for attempt in range(30):
            try:
                response = requests.get(uri, timeout=5)
                response.raise_for_status()
                items = response.json().get('items') or []
                if items:
                    data = items[0]
                    break
                last_err = "setting collection is empty"
            except Exception as e:  # connection refused / timeout / 5xx
                last_err = e
            time.sleep(1.0)
        if data is None:
            logging.error(
                f"loadConfig: PocketBase not ready after retries ({last_err}); "
                f"falling back to built-in defaults")

        data = data or {}

        def _num(key, default, cast=float):
            try:
                value = data.get(key)
                if value is None or value == "":
                    return default
                return cast(value)
            except (TypeError, ValueError):
                return default

        if data.get('isRfid'):
            self.ip_relay, self.ip_port, self.relayN1, self.relayN2 = (
                data.get('rfidip', '').strip(), data.get('rfidport'),
                data.get('rl1'), data.get('rl2'))
        if data.get('rl1'):
            self.relayN1 = 1
        if data.get('rl2'):
            self.relayN2 = 2

        cfg = {
            'score': _num('score', 0.6),
            'padding': _num('padding', 40, int),
            'quality': _num('quality', 100, int),
            'hscore': _num('hscore', 0.6),
            'simscore': _num('simscore', 0.6),
            'port': data.get('port') or '8003',
            'isRegionMode': bool(data.get('isregion', False)),
            'isRelay': bool(data.get('isRfid', False)),
            'iou': iou,
            # New accuracy gates (see change_to_database.txt): a zeroed
            # column means "not configured", see resolve_setting_gates.
            **resolve_setting_gates(data),
        }
        logging.info(
            f"Config: score={cfg['score']} simscore={cfg['simscore']} "
            f"marginMin={cfg['marginMin']} voteObs={cfg['voteObs']} "
            f"minBlur={cfg['minBlur']} maxYaw={cfg['maxYaw']} "
            f"minFacePx={cfg['minFacePx']}")
        return cfg

    def load_image_searcher_model(self):
        """Load the ResNet50 search model once and cache it"""
        if self._image_searcher_model is None:
            with self._image_searcher_lock:
                if self._image_searcher_model is None:
                    model = resnet50(weights=None)  # don't load default
                    state_dict = torch.load(
                        self.LOCAL_WEIGHTS, map_location=self.device)
                    model.load_state_dict(state_dict)
                    model = torch.nn.Sequential(*(list(model.children())[:-1]))
                    model.eval().to(self.device)
                    self._image_searcher_model = model
        return self._image_searcher_model

    def get_embedding(self, img_path, model=None):
        model = model or self.load_image_searcher_model()
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = model(img_t)
        features = features.view(features.size(0), -1).cpu().numpy().flatten()
        return features / np.linalg.norm(features)

    def precompute_embeddings(self, model=None, folder_path=None):
        """Incrementally embed only files missing from the saved index"""
        model = model or self.load_image_searcher_model()
        folder_path = folder_path or self.FOLDER_PATH
        if os.path.exists(self.EMBEDDING_FILE) and os.path.exists(self.FILENAMES_FILE):
            embeddings, filenames = self.load_embeddings()
            embeddings = embeddings.tolist()
        else:
            embeddings, filenames = [], []

        known = set(filenames)
        new_count = 0
        for fname in sorted(os.listdir(folder_path)):
            if not fname.lower().endswith(self.IMG_EXTENSIONS):
                continue
            if fname in known:
                continue
            emb = self.get_embedding(os.path.join(folder_path, fname), model)
            embeddings.append(emb)
            filenames.append(fname)
            new_count += 1

        if not embeddings:
            return np.empty((0, 2048), dtype=np.float32), []

        embeddings = np.array(embeddings)
        np.save(self.EMBEDDING_FILE, embeddings)
        with open(self.FILENAMES_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(filenames))
        logging.info(
            f"Image-search index updated: {new_count} new, {len(filenames)} total")
        return embeddings, filenames

    def load_embeddings(self):
        embeddings = np.load(self.EMBEDDING_FILE)
        with open(self.FILENAMES_FILE, "r", encoding='utf-8') as f:
            filenames = f.read().splitlines()
        logging.info(f"Loaded {len(filenames)} embeddings from disk")
        return embeddings, filenames

    def find_similar_images(self, query_embedding, embeddings, filenames, top_k=10):
        # All stored embeddings are L2-normalized at creation time
        # (get_embedding), so cosine similarity is just the dot product.
        sims = np.asarray(embeddings, dtype=np.float32) @ np.asarray(
            query_embedding, dtype=np.float32)
        if sims.size and np.max(sims) > SIMILARITY_THRESHOLD:
            sorted_indices = np.argsort(sims)[::-1]
            results = [(filenames[i], float(sims[i])) for i in sorted_indices[:top_k]]
            return results
        return []

    def create_yolo_instance(self):
        """Create a fresh YOLO instance so per-camera trackers don't share state"""
        if self.device == 'cpu' and self.checkOpenVino():
            logging.info('Loadin openvino')
            return YOLO('models/yolov8n_openvino_model',
                        task='detect', verbose=False)
        logging.info('Loadin onnx/pt')
        model = YOLO(self.MODEL_PATH, task='detect', verbose=False)
        if self.fileEx != 'onnx':
            model.eval()
        return model

    def create_face_instance(self):
        """Create a dedicated InsightFace session for one camera thread.

        Opt-in: set MAX_FACE_SESSIONS>0 to enable (benchmark.py first -
        concurrent sessions hurt throughput on some GPUs). Returns None
        when the cap is reached or creation failed (e.g. VRAM); callers
        then fall back to the shared handler guarded by face_lock. A
        failed creation disables further attempts so a full GPU doesn't
        get hammered with retries.
        """
        if self._face_sessions_disabled or self._face_sessions_created >= MAX_FACE_SESSIONS:
            return None
        try:
            handler = FaceAnalysis(
                'antelopev2',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider'],
                root='.'
            )
            prepare_face_handler(handler)
            self._face_sessions_created += 1
            logging.info(
                f"Dedicated face session {self._face_sessions_created}/{MAX_FACE_SESSIONS} loaded")
            return handler
        except Exception as e:
            self._face_sessions_disabled = True
            logging.warning(
                f"Could not create dedicated face session, sharing the global one: {e}")
            return None

    def _load_models(self):
        """Load YOLO and face recognition models"""
        try:
            logging.info(f"Loading models...")

            # Load face handler
            self.face_handler = FaceAnalysis(
                'antelopev2',
                providers= ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device=='cuda' else ['CPUExecutionProvider'],
                root='.'
            )
            prepare_face_handler(self.face_handler)

            # Shared one-shot YOLO instance; each camera creates its own via
            # create_yolo_instance so tracker state never mixes across cameras
            self.model = self.create_yolo_instance()

            logging.info('Models loaded successfully.')

        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise

    def start(self):
        self.process = subprocess.Popen(
            ["pocketbase", "serve", "--http=0.0.0.0:8091"], creationflags=subprocess.CREATE_NO_WINDOW)
        logging.info(f"PocketBase stater {self.process.pid}")

    def load_db(self):
        """Load known faces from database"""
        try:
            known_names = load_embeddings_from_db()
            logging.info(
                f"Loaded {len(known_names)} known faces from database")
            return known_names
        except Exception as e:
            logging.error(f"Failed to load database: {e}")
            return {}

    def _build_embedding_index(self):
        """Pre-build flat numpy matrix for fast batch cosine similarity.

        Rows are grouped by person so ``recognize_face`` can reduce to one
        score per identity with ``np.maximum.reduceat`` (no per-frame
        Python loop over every reference image).

        Thread-safe: acquires _index_lock to serialise rebuilds.  The swap
        itself is a single tuple assignment so readers always see a
        consistent (matrix, person_labels, person_starts) snapshot.
        """
        all_embeddings = []
        person_labels = []
        person_starts = []

        for name, person_data in self.known_names.items():
            embs = person_data.get('embeddings', [])
            if not embs:
                continue
            # 7-tuple; index 6 is the per-person threshold override
            # (0/None = use the global simscore).
            person_labels.append((
                name,
                person_data.get('age', 'None'),
                person_data.get('gender', 'None'),
                person_data.get('role', ''),
                person_data.get('socialnumber', ''),
                person_data.get('id', ''),
                person_data.get('simThreshold'),
            ))
            person_starts.append(len(all_embeddings))
            all_embeddings.extend(embs)

        if all_embeddings:
            matrix = np.array(all_embeddings, dtype=np.float32)
            # Normalize all rows once
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            matrix = matrix / norms
            starts = np.asarray(person_starts, dtype=np.int64)
        else:
            matrix = np.empty((0, 512), dtype=np.float32)
            starts = np.empty((0,), dtype=np.int64)

        # Single atomic swap under lock so concurrent rebuilds (e.g. two
        # registration API calls) don't race, while recognition threads
        # read the tuple without locking (they see either old or new,
        # never half-built).
        with self._index_lock:
            self.embedding_index = (matrix, person_labels, starts)
        logging.info(
            f"Embedding index built: {len(person_labels)} persons / "
            f"{len(all_embeddings)} vectors")

    def refresh_person(self, name):
        """Incrementally refresh a single person instead of a full DB reload.

        Avoids re-fetching up to 1000 records over HTTP on every known-person
        insert; only the one changed record is loaded, then the in-memory
        index is rebuilt (cheap, no network).
        """
        person = load_person_from_db(name)
        if person is None:
            logging.warning(f"refresh_person: no record found for '{name}'")
            return
        with self._index_lock:
            self.known_names[name] = person[name]
        self._build_embedding_index()
        logging.info(f"Refreshed person '{name}' in CCTV monitor")

    def remove_person(self, name):
        """Evict a deleted person from memory and rebuild the index.

        ``refresh_person`` cannot do this: it early-returns when the DB
        record is gone, leaving the deleted person recognised until
        restart.  Thread-safe via the index lock + atomic tuple swap.
        """
        with self._index_lock:
            existed = self.known_names.pop(name, None) is not None
        if existed:
            self._build_embedding_index()
            logging.info(f"Removed person '{name}' from CCTV monitor")
        else:
            logging.debug(f"remove_person: '{name}' not in memory")

    async def graceful_shutdown(self):
        """Gracefully shutdown the system"""
        logging.info("Initiating graceful shutdown...")

        # Signal shutdown
        self._shutdown_event.set()

        # Clean up GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Clean up thread pool
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)

        # Garbage collection
        gc.collect()

        # Terminate subprocess if exists
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        logging.info("Cleanup complete.")


class CameraManager:
    def __init__(self, source, config: CCtvMonitor,camera_id):
        self.source = source
        self.config = config
        self.camera_id = camera_id

        # ---------- STATE ----------
        self.running = False
        self.client_count = 0
        self.client_lock = threading.Lock()

        # ---------- THREADS ----------
        self.capture_thread = None
        self.process_thread = None
        self.recognition_thread = None
        self.stop_event = threading.Event()

        # ========== TRIPLE BUFFERS ==========
        # Writers publish immutable frames/bytes and advance the read index
        # under a lock; readers fetch read_idx once and keep a valid reference.
        self.capture_buffer = [None, None, None]  # raw frames from camera
        self.display_buffer = [None, None, None]  # encoded JPEG bytes
        self.capture_read_idx = 0
        self.display_read_idx = 0
        
        self.capture_version = 0
        self.display_version = 0
        self._capture_lock = threading.Lock()
        self._display_lock = threading.Lock()
        
        # ========== OPTIMIZED QUEUES ==========
        self.frame_queue = queue.Queue(maxsize=2)
        # Recognition is fed by a latest-per-track slot map instead of a
        # bounded queue: with a queue, N people all pushing every frame
        # overflowed maxsize=5 and the tracks that lost the race were
        # never recognised at all.  A map keeps exactly one pending crop
        # per track, so no person can starve another.
        self._pending_lock = threading.Lock()
        self._pending_tracks = {}
        self._pending_event = threading.Event()
        
        # ---------- DATA ----------
        self._processed_tracks_lock = threading.Lock()
        self.processed_tracks = set()
        self.face_info = {}
        self.face_info_lock = threading.Lock()
        self.embedding_cache = {}
        self._cache_lock = threading.Lock()
        # Per-track observation buffer used to vote on an identity before
        # committing it (see _fuse_observations / _commit_track).
        self.track_obs = {}
        self._track_obs_lock = threading.Lock()
        self.last_seen = {}  # track_id -> timestamp, for pruning stale state
        self._last_prune = 0.0
        self._jpeg_params = [cv2.IMWRITE_JPEG_QUALITY,
                             70, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
        self.model = None  # per-camera YOLO, created on start()
        # Dedicated InsightFace session for this camera (falls back to the
        # shared config.face_handler behind face_lock when None)
        self.face_handler = None
        
        if self.config.isRegionMode:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
            self.k = []

    def start(self):
        self.running = True
        self.stop_event.clear()

        if self.model is None:
            # Own YOLO instance per camera: bytetrack state must not be shared
            self.model = self.config.create_yolo_instance()

        if self.face_handler is None:
            # Own InsightFace session per camera so recognition never queues
            # behind other cameras' GPU work on the global face_lock
            self.face_handler = self.config.create_face_instance()

        self.capture_thread = threading.Thread(
            target=self.generate_frames, args=[self.camera_id,self.source],daemon=True
        )
        self.process_thread = threading.Thread(
            target=self.process_frame, daemon=True
        )
        self.recognition_thread= threading.Thread(
                target=self.recognition_worker,
                daemon=True,
            )
       
        self.capture_thread.start()
        self.process_thread.start()
        self.recognition_thread.start()
        

    def stop(self):
        self.running = False
        self.stop_event.set()

    def add_client(self):
        with self.client_lock:
            self.client_count += 1

            if self.client_count == 1:
                self.start()

    def remove_client(self):
        with self.client_lock:
            self.client_count -= 1

            if self.client_count == 0:
                self.stop()

    def has_clients(self):
        with self.client_lock:
            return self.client_count > 0

            
        
    def sendFrames(self):
        # Frames arrive pre-encoded from process_frame; every client shares
        # the same bytes instead of encoding per client.
        last_version = -1
        while self.running:
            current_version = self.display_version
            if current_version == last_version:
                time.sleep(0.003)
                continue
            last_version = current_version
            jpeg_bytes = self.display_buffer[self.display_read_idx]

            if jpeg_bytes is None:
                time.sleep(0.003)
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg_bytes
                + b"\r\n"
            )
    

    def generate_frames(self, camera_idx, source):
        """Generate frames from a specific camera feed"""
        if not self.is_connection_alive(source):
            logging.warning(f"[Camera {camera_idx}] Connection not available")
            return

        counter = 0
        region_masks = None
        combined_mask = None
        regions = self.load_regions(soruce=source) if self.config.isRegionMode else None
        if self.config.isRegionMode and not hasattr(self, 'k'):
            self.k = []

        fresh = FreshestFrame(source)

        try:
            while self.running and not self.stop_event.is_set():

                success, frame = fresh.read()
                counter += 1
                

                if frame is None:
                    continue

                # Region masks only depend on frame size, build them once
                if self.config.isRegionMode and region_masks is None:
                    # If this camera has no region defined in regions.json,
                    # auto-create one that covers the entire frame and
                    # persist it so it isn't recreated on every restart.
                    if not regions:
                        regions = self.create_default_region(source, frame.shape)
                    region_masks = self.generate_region_masks(frame.shape, regions)
                    combined_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    for mask in region_masks.values():
                        combined_mask = cv2.bitwise_or(combined_mask, mask)

                with self._capture_lock:
                    write_idx = (self.capture_read_idx + 1) % len(self.capture_buffer)
                    self.capture_buffer[write_idx] = frame
                    self.capture_read_idx = write_idx
                    self.capture_version += 1
                

               
                

                try:
                    self.frame_queue.put_nowait(
                        (f'/rt{camera_idx}', counter, regions, region_masks, combined_mask))
                except queue.Full:
                    pass

        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")
        finally:
            logging.info("Releasing camera resources")
            fresh.release()

    def is_connection_alive(self, source):
        """Check if network connection to source is alive"""
        return _is_connection_alive(source)

    def process_frame(self):
        last_capture_version = -1
        """Process a single frame for object detection and face recognition"""
        while self.running :
            try:

                item = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                if not self.running:
                    break
                continue

            if item is None:
                logging.info("process_frame shutdown signal received")
                break
            path, counter, regions, region_masks, combined_mask = item
            current_capture_version = self.capture_version

            # Skip if same frame
            if current_capture_version == last_capture_version:
                continue

            last_capture_version = current_capture_version

            # Read from stable read buffer
            frame = self.capture_buffer[self.capture_read_idx]
            if frame is None or frame.size == 0:
                continue

            try:
                now = start_time = time.time()
                processed_frame = frame.copy()
                if self.config.isRegionMode:
                    # Masks were built once in the capture thread
                    masked_frame = cv2.bitwise_and(
                        processed_frame, processed_frame, mask=combined_mask)
                    self.k.clear()
                    current_regions = []

                # Run YOLO detection on this camera's own instance
                _t0 = time.perf_counter()
                results = self.model.track(
                    masked_frame if self.config.isRegionMode else processed_frame,
                    classes=[0],  # Person class
                    iou=self.config.iou,
                    tracker="bytetrack.yaml",
                    persist=True,
                    device=self.config.device,
                    conf=self.config.hscore,
                )
                PERF_STATS.add(f"cam{self.camera_id}:yolo", time.perf_counter() - _t0)

                for res in results:
                    if res.boxes.id is None:
                        continue
                    for i in range(len(res.boxes.xyxy)):
                        x1, y1, x2, y2 = res.boxes.xyxy[i].int().tolist()
                        region_data = None
                        if self.config.isRegionMode:
                            region_name = self.get_detection_region(
                                (x1, y1, x2, y2), region_masks)
                            if region_name and region_name in regions:
                                region_data = regions[region_name]
                                if region_data not in current_regions:
                                    current_regions.append(region_data)

                        # Get tracking ID
                        track_id = int(res.boxes.id[i])
                        self.last_seen[track_id] = now

                        # Crop human region
                        human_crop = masked_frame[y1:y2,
                                                  x1:x2] if self.config.isRegionMode else processed_frame[y1:y2, x1:x2]
                        if human_crop.size == 0:
                            continue

                        # Draw bounding box
                        # cv2.rectangle(processed_frame, (x1, y1),
                        #               (x2, y2), (0, 255, 0), 2)

                        # Queue for recognition if not yet processed or
                        # cooldown elapsed
                        with self._processed_tracks_lock:
                            already_processed = track_id in self.processed_tracks
                        if not already_processed:
                            self._queue_track(
                                track_id,
                                (path, track_id, human_crop.copy(),
                                 region_data))

                        # Get face info
                        with self.face_info_lock:
                            info = self.face_info.get(
                                track_id,
                                {
                                    'name': "Unknown",
                                    'score': 0,
                                    'bbox': None,
                                    'gender': 'None',
                                    'age': 'None',
                                    'role': '',
                                    'socialnumber': ''
                                }
                            )

                        # Create label
                        label = f"{info['name']} ID:{track_id}"
                        face_bbox = info['bbox']

                        if face_bbox:
                            fx1, fy1, fx2, fy2 = face_bbox
                            # cv2.rectangle(
                            #     processed_frame,
                            #     (x1 + fx1, y1 + fy1),
                            #     (x1 + fx2, y1 + fy2),
                            #     (0, 0, 255), 2
                            # )
                            cv2.putText(
                                processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                            )

                        else:
                            cv2.putText(
                                processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                            )

                if now - self._last_prune > STALE_TRACK_TTL:
                    self._prune_stale_tracks(now)
                    self._last_prune = now

                # Calculate and display FPS
                if self.config.isRegionMode:
                    self.k = current_regions
                    self.onDisplay(self.k, processed_frame)
                    display_frame = self.draw_regions_on_frame(
                        processed_frame, regions)
                else:
                    display_frame = processed_frame

                try:
                    fps = 1.0 / (time.time() - start_time)
                except ZeroDivisionError:
                    fps = 30
                cv2.putText(
                    display_frame, f"FPS: {fps:.2f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                # Encode once here; all streaming clients share these bytes
                _t0 = time.perf_counter()
                _, jpeg = cv2.imencode(".jpg", display_frame, self._jpeg_params)
                PERF_STATS.add(f"cam{self.camera_id}:jpeg_encode", time.perf_counter() - _t0)
                with self._display_lock:
                    write_idx = (self.display_read_idx + 1) % len(self.display_buffer)
                    self.display_buffer[write_idx] = jpeg.tobytes()
                    self.display_read_idx = write_idx
                    self.display_version += 1
                PERF_STATS.add(f"cam{self.camera_id}:frame_total", time.time() - start_time)

            except Exception as e:
                logging.error(f"Error processing frame: {e}")

    def _prune_stale_tracks(self, now):
        """Drop state for track IDs that have not been seen recently"""
        stale = [tid for tid, ts in self.last_seen.items()
                 if now - ts > STALE_TRACK_TTL]
        if not stale:
            return
        with self.face_info_lock:
            for tid in stale:
                self.face_info.pop(tid, None)
        with self._processed_tracks_lock:
            for tid in stale:
                self.processed_tracks.discard(tid)
        with self._cache_lock:
            for tid in stale:
                self.embedding_cache.pop(tid, None)
        with self._track_obs_lock:
            for tid in stale:
                self.track_obs.pop(tid, None)
        for tid in stale:
            self.last_seen.pop(tid, None)

    def _queue_track(self, track_id, item):
        """Store the newest pending crop for *track_id* (never blocks).

        Overwriting is intentional: the worker only needs the latest frame
        of a person, so a slow recognition pass can never make this
        producer drop anybody.
        """
        with self._pending_lock:
            self._pending_tracks[track_id] = item
        self._pending_event.set()

    def _take_pending(self, timeout=0.05):
        """Wait up to *timeout* and return every pending track's newest item."""
        self._pending_event.wait(timeout)
        self._pending_event.clear()
        with self._pending_lock:
            items = list(self._pending_tracks.values())
            self._pending_tracks.clear()
        return items

    def recognition_worker(self):
        """Background worker for face recognition with batch queue draining"""
        logging.info(f"Recognition worker started for cam{self.camera_id}")

        while not self.stop_event.is_set():
            try:
                pending = self._take_pending(0.05)
                if not pending:
                    continue

                for path, track_id, face_img, region_data in pending:
                    now = time.time()

                    # Per-track throttle + observation buffer.  Unlike the
                    # old code (one recognition per track, ever) we collect
                    # several quality observations and vote on them, so a
                    # single blurred/occluded frame cannot decide an
                    # identity.  Only this worker thread mutates `state`;
                    # the lock only protects the map itself.
                    with self._track_obs_lock:
                        state = self.track_obs.get(track_id)
                        if state is None:
                            state = {'obs': [], 'attempts': 0,
                                     'last_attempt': 0.0, 'committed': False}
                            self.track_obs[track_id] = state
                    if state['committed']:
                        continue
                    if now - state['last_attempt'] < OBS_INTERVAL:
                        PERF_STATS.add(
                            f"cam{self.camera_id}:skip_throttled", 0.0)
                        continue
                    state['last_attempt'] = now
                    state['attempts'] += 1

                    _t0 = time.perf_counter()
                    if self.face_handler is not None:
                        # Dedicated session: no cross-camera lock needed
                        faces = self.face_handler.get(face_img)
                        PERF_STATS.add(
                            f"cam{self.camera_id}:face", time.perf_counter() - _t0)
                    else:
                        # Shared fallback: measure lock wait separately so
                        # [perf] logs show if contention is the bottleneck
                        with self.config.face_lock:
                            PERF_STATS.add(
                                f"cam{self.camera_id}:lock_wait",
                                time.perf_counter() - _t0)
                            _t1 = time.perf_counter()
                            faces = self.config.face_handler.get(face_img)
                        PERF_STATS.add(
                            f"cam{self.camera_id}:face", time.perf_counter() - _t1)

                    if not faces:
                        self.update_face_info(
                            track_id, "Unknown", 0.0, 'None', 'None', '', '', None
                        )
                        self._give_up_if_exhausted(
                            track_id, state, path, face_img, region_data,
                            None, 0.0)
                        continue

                    # A person crop can contain background faces;
                    # the largest one is the tracked subject.
                    if len(faces) > 1:
                        logging.debug(
                            f"cam{self.camera_id}: {len(faces)} faces in "
                            f"track {track_id} crop, selecting largest")
                    face = select_primary_face(faces)

                    x1, y1, x2, y2 = map(int, face.bbox)
                    if min(x2 - x1, y2 - y1) < get_min_face_px(self.config):
                        PERF_STATS.add(
                            f"cam{self.camera_id}:skip_small_face", 0.0)
                        self.update_face_info(
                            track_id, "Unknown", 0.0, 'None', 'None', '', '', None
                        )
                        self._give_up_if_exhausted(
                            track_id, state, path, face_img, region_data,
                            (x1, y1, x2, y2), 0.0)
                        continue

                    gender = 'female' if face.gender == 0 else 'male'
                    age = face.age

                    det_score = float(face.det_score)
                    if det_score <= self.config.score:
                        self.update_face_info(
                            track_id, "Unknown", 0.0, 'None', 'None', '', '', None
                        )
                        self._give_up_if_exhausted(
                            track_id, state, path, face_img, region_data,
                            (x1, y1, x2, y2), det_score)
                        continue

                    h_img, w_img = face_img.shape[:2]
                    face_crop = face_img[max(y1, 0):min(y2, h_img),
                                         max(x1, 0):min(x2, w_img)]
                    blur = face_blur_score(face_crop)
                    yaw = face_yaw(face)

                    name, sim, margin, gender, age, role, socialnumber, pid = (
                        self.recognize_face(face.embedding, gender, age))

                    self.update_face_info(
                        track_id, name, sim, gender, age, role, socialnumber,
                        (x1, y1, x2, y2), margin=margin, person_id=pid,
                        blur=blur, yaw=yaw, obs=len(state['obs'])
                    )
                    with self._cache_lock:
                        self.embedding_cache[track_id] = face.embedding

                    # Quality gate: only sharp, frontal, confident frames
                    # count as evidence for the vote.
                    min_blur = float(getattr(
                        self.config, 'minBlur', DEFAULT_MIN_BLUR))
                    max_yaw = float(getattr(
                        self.config, 'maxYaw', DEFAULT_MAX_YAW))
                    if blur < min_blur or abs(yaw) > max_yaw:
                        PERF_STATS.add(f"cam{self.camera_id}:skip_quality", 0.0)
                        logging.debug(
                            f"cam{self.camera_id}: track {track_id} rejected "
                            f"(blur={blur:.1f}<{min_blur} yaw={yaw:.1f}"
                            f" vs {max_yaw})")
                        self._give_up_if_exhausted(
                            track_id, state, path, face_img, region_data,
                            (x1, y1, x2, y2), det_score)
                        continue

                    state['obs'].append({
                        'pid': pid, 'name': name, 'sim': float(sim),
                        'margin': float(margin), 'gender': gender, 'age': age,
                        'role': role, 'socialnumber': socialnumber,
                        'det': det_score, 'blur': blur, 'yaw': yaw,
                    })

                    fused = self._fuse_observations(state['obs'])
                    if fused is None:
                        continue
                    self.update_face_info(
                        track_id, fused['name'], fused['sim'], fused['gender'],
                        fused['age'], fused['role'], fused['socialnumber'],
                        (x1, y1, x2, y2), margin=fused['margin'],
                        person_id=fused['pid'], blur=blur, yaw=yaw,
                        obs=fused['n']
                    )

                    if not self._fusion_is_decisive(fused):
                        self._give_up_if_exhausted(
                            track_id, state, path, face_img, region_data,
                            (x1, y1, x2, y2), det_score)
                        continue

                    self._commit_track(
                        track_id, state, fused, path, face_img, region_data,
                        det_score, (x1, y1, x2, y2))

            except queue.Empty:
                continue
        logging.info(f"Recognition worker stopped for cam{self.camera_id}")

    # ------------------------------------------------------------------
    #  Temporal fusion: several observations -> one committed identity
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse_observations(obs):
        """Majority-vote the per-frame identity decisions of one track.

        Returns None when there is nothing to fuse.  ``sim`` is the mean
        similarity across the winning identity's frames and ``margin`` the
        gap to the best competing identity (or a large value when nobody
        competed).
        """
        if not obs:
            return None
        groups = {}
        for o in obs:
            key = o.get('pid') or ''
            g = groups.get(key)
            if g is None:
                g = groups[key] = {'n': 0, 'sum': 0.0, 'first': o}
            g['n'] += 1
            g['sum'] += o['sim']

        winner_key = max(groups, key=lambda k: (groups[k]['n'], groups[k]['sum']))
        winner = groups[winner_key]
        others = [g for k, g in groups.items() if k != winner_key]
        other_sim = max((g['sum'] / g['n'] for g in others), default=-1.0)
        sim = winner['sum'] / winner['n']
        first = winner['first']
        return {
            'pid': first.get('pid', ''), 'name': first['name'],
            'gender': first['gender'], 'age': first['age'],
            'role': first['role'], 'socialnumber': first['socialnumber'],
            'sim': float(sim), 'margin': float(sim - other_sim),
            'n': winner['n'], 'total': len(obs),
            'ratio': winner['n'] / len(obs),
            'blur': first.get('blur', 0.0), 'yaw': first.get('yaw', 0.0),
        }

    @staticmethod
    def _min_observations(vote_obs: int) -> int:
        """Observations needed before a unanimous track may be trusted."""
        return max(2, min(int(vote_obs), 3))

    def _fusion_is_decisive(self, fused) -> bool:
        """True when enough independent observations agree on one identity."""
        vote_obs = int(getattr(self.config, 'voteObs', DEFAULT_VOTE_OBS))
        majority = float(getattr(
            self.config, 'voteMajority', DEFAULT_VOTE_MAJORITY))
        if fused['n'] < self._min_observations(vote_obs):
            return False
        if fused['ratio'] < majority:
            return False
        if fused['pid']:
            # A known identity must also clear the similarity and margin
            # gates: the mean can dip below the per-frame threshold when
            # later frames are worse, and a tight margin means two known
            # people were fighting over the same frames.
            if fused['sim'] < self.config.simscore:
                return False
            if fused['margin'] < float(getattr(
                    self.config, 'marginMin', DEFAULT_MARGIN_MIN)):
                return False
        return True

    def _give_up_if_exhausted(self, track_id, state, path, face_img,
                              region_data, bbox, det_score):
        """Settle a track that never became decisive.

        Without this, a track whose frames are all rejected (blurry,
        extreme pose) would keep the recognition queue busy until it went
        stale.  The result is recorded as 'unknown' rather than as the
        most-voted identity, because an undecided track must not assert an
        identity.
        """
        if state['committed']:
            return
        vote_obs = int(getattr(self.config, 'voteObs', DEFAULT_VOTE_OBS))
        cap = max(self._min_observations(vote_obs) * 4, vote_obs * 2)
        if state['attempts'] < cap:
            return

        fused = self._fuse_observations(state['obs'])
        if fused is None or not self._fusion_is_decisive(fused):
            logging.info(
                f"cam{self.camera_id}: track {track_id} undecided after "
                f"{state['attempts']} attempts / {len(state['obs'])} "
                f"observations -> recorded as unknown")
            fused = {
                'pid': '', 'name': 'unknown', 'gender': 'None', 'age': 'None',
                'role': '', 'socialnumber': '', 'sim': 0.0, 'margin': 0.0,
                'n': 0, 'total': len(state['obs']), 'ratio': 0.0,
                'blur': 0.0, 'yaw': 0.0,
            }
        # Only log when we actually saw a face (det_score > 0); otherwise
        # a track that never produced one would leave a 0-confidence row.
        log_bbox = bbox if det_score and det_score > 0 else None
        self._commit_track(
            track_id, state, fused, path, face_img, region_data,
            det_score, log_bbox)

    def _commit_track(self, track_id, state, fused, path, face_img,
                      region_data, det_score, bbox):
        """Log the settled identity once and stop re-queueing this track."""
        if state['committed']:
            return
        state['committed'] = True
        with self._processed_tracks_lock:
            self.processed_tracks.add(track_id)

        if bbox is None:
            return

        try:
            
            x1, y1, x2, y2 = bbox
            height_f, width_f = face_img.shape[:2]
            padding = 40
            fx1_padded = max(x1 - padding, 0)
            fy1_padded = max(y1 - padding, 0)
            fx2_padded = min(x2 + padding, width_f)
            fy2_padded = min(y2 + padding, height_f)

            cropped_face = face_img[fy1_padded:fy2_padded,
                                    fx1_padded:fx2_padded]

            read_idx = self.capture_read_idx
            current_full_frame = self.capture_buffer[read_idx]
            full_frame_copy = (current_full_frame.copy()
                               if current_full_frame is not None else None)
            # DB insert submitted to the dedicated DbWorker so it never
            # blocks the recognition or video pipeline even if PocketBase
            # is slow.
            submit_db_task(
                insertToDb, fused['name'], full_frame_copy,
                cropped_face, face_img, det_score,
                track_id, fused['gender'], fused['age'], fused['role'],
                fused['socialnumber'], path,
                self.config.quality, region_data,
                self.config.isRelay, self.config.isRegionMode,
                self.config.ip_relay, self.config.ip_port,
                self.config.relayN1, self.config.relayN2,
                sim=fused['sim'], margin=fused['margin'],
                obs_count=fused['n'], person_id=fused['pid'],
                blur=fused['blur'], yaw=fused['yaw'], det_score=det_score)
        except Exception as e:
            logging.error(f"Error queueing DB insert: {e}")

    def recognize_face(self, embedding, fgender, fage):
        """Best identity for a single frame, plus an ambiguity margin.

        Returns ``(name, sim, margin, gender, age, role, socialnumber,
        person_id)``.

        Reference rows are grouped by person at index-build time, so the
        per-person score is a max over that person's images computed with
        ``np.maximum.reduceat`` - no per-frame Python loop over the whole
        gallery.  ``margin`` is best-minus-second *distinct person*; a
        small value means two known people were fighting over this frame,
        which is what the caller uses to refuse a confident-looking match.

        The embedding_index is an atomic tuple swap; readers never need a
        lock because tuple reads are atomic and writers publish a complete
        new tuple.
        """
        matrix, person_labels, person_starts = self.config.embedding_index
        if matrix.shape[0] == 0 or not person_labels:
            return "unknown", 0.0, 1.0, fgender, fage, '', '', ''

        query = np.asarray(embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        sims = matrix @ query
        person_sims = np.maximum.reduceat(sims, person_starts)

        order = np.argsort(person_sims)[::-1]
        best_i = int(order[0])
        best_sim = float(person_sims[best_i])
        second_sim = (float(person_sims[int(order[1])])
                      if order.size > 1 else -1.0)
        margin = best_sim - second_sim

        name, age, gender, role, socialnumber, pid, override = (
            person_labels[best_i])
        thr = float(override) if override else float(self.config.simscore)
        if thr <= 0:
            thr = float(self.config.simscore)

        # Debug-level diagnosis for "why wasn't this face recognised?"
        # (enable with logging level DEBUG; no per-frame INFO spam).
        logging.debug(
            f"recognize_face: best={best_sim:.4f} ({name}) "
            f"second={second_sim:.4f} margin={margin:.4f} thr={thr} "
            f"persons={len(person_labels)} refs={matrix.shape[0]}")

        if best_sim >= thr:
            return name, best_sim, margin, gender, age, role, socialnumber, pid

        return "unknown", max(best_sim, 0.0), margin, fgender, fage, '', '', ''

    def update_face_info(self, track_id, name, score, gender, age, role,
                         socialnumber, bbox=None, margin=0.0, person_id='',
                         blur=0.0, yaw=0.0, obs=0):
        """Thread-safe update of face information"""
        with self.face_info_lock:
            self.face_info[track_id] = {
                'name': name,
                'bbox': bbox,
                'last_update': time.time(),
                'score': score,
                'gender': gender,
                'age': age,
                'role': role,
                'socialnumber': socialnumber,
                'margin': margin,
                'person_id': person_id,
                'blur': blur,
                'yaw': yaw,
                'obs': obs,
            }

    def release_resources(self, role=False):
        if not self.running:
            return

        self.running = False
        logging.info("Camera pipeline stopped")

    def load_regions(self, soruce, file_path='regions.json',):
        url = urlparse(soruce).hostname
        """Load regions from JSON file"""
        try:
            with open(file_path, 'r') as f:
                datas = json.load(f)
                for data in datas:
                    if url == data['ip']:
                        return data.get('regions', {})
                    else:
                        pass

        except Exception as e:
            logging.error(f"Error loading regions: {e}")
            return {}

    def create_default_region(self, source, frame_shape, file_path='regions.json'):
        """Create (and persist) a region covering the entire frame."""
        h, w = frame_shape[:2]
        region = {
            "auto": {
                "id": "auto",
                "name": "auto",
                "description": "Auto-created full-screen region",
                "points": [
                    [0.0, 0.0],
                    [float(w), 0.0],
                    [float(w), float(h)],
                    [0.0, float(h)],
                    [0.0, 0.0]
                ],
                "shape_type": "polygon",
                "color": "red",
                "created": datetime.datetime.now().isoformat(),
                "ip": urlparse(source).hostname,
                "relay_ip": None,
                "relay_number": None
            }
        }
        self.save_regions(source, region, file_path)
        return region

    def save_regions(self, source, new_regions, file_path='regions.json'):
        """Persist regions for a camera into regions.json (merged by IP)."""
        url = urlparse(source).hostname
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    datas = json.load(f)
            else:
                datas = []

            found = False
            for data in datas:
                if data.get('ip') == url:
                    data.setdefault('regions', {}).update(new_regions)
                    found = True
                    break
            if not found:
                datas.append({'ip': url, 'regions': dict(new_regions)})

            with open(file_path, 'w') as f:
                json.dump(datas, f, indent=2)
            logging.info(f"Auto-added region for camera {url}")
        except Exception as e:
            logging.error(f"Error saving regions for {url}: {e}")

    def draw_regions_on_frame(self, frame, regions):
        """Draw region boundaries on frame"""
        overlay = frame.copy()

        for region_name, region_data in regions.items():
            points = region_data.get('points', [])
            color_name = region_data.get('color', 'red')
            shape_type = region_data.get('shape_type', 'polygon')

            # Convert color name to BGR
            color_map = {
                'red': (0, 0, 255), 'blue': (255, 0, 0), 'green': (0, 255, 0),
                'yellow': (0, 255, 255), 'purple': (128, 0, 128),
                'orange': (0, 165, 255), 'cyan': (255, 255, 0), 'magenta': (255, 0, 255)
            }
            color = color_map.get(color_name, (0, 0, 255))

            if shape_type == 'polygon' and len(points) > 2:
                pts = np.array(points, dtype=np.int32)
                cv2.polylines(overlay, [pts], True, color, 2)

            elif shape_type == 'rectangle' and len(points) == 4:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[2][0]), int(points[2][1])
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

            elif shape_type == 'line' and len(points) == 2:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                cv2.line(overlay, (x1, y1), (x2, y2), color, 2)

            # Add region label
            if points:
                center_x = int(sum(p[0] for p in points) / len(points))
                center_y = int(sum(p[1] for p in points) / len(points))

                text = f"{region_name} (ID: {region_data.get('id', 'N/A')})"
                text_size = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

        return overlay

    def get_detection_region(self, detection_box, region_masks):

        cx = int((detection_box[0] + detection_box[2]) / 2)
        cy = int((detection_box[1] + detection_box[3]) / 2)
        for region_name, mask in region_masks.items():

            if cy < mask.shape[0] and cx < mask.shape[1] and mask[cy, cx] > 0:
                return region_name  # First match wins
        return None

    def generate_region_masks(self, frame_shape, regions):
        """Create binary masks for each region (once)"""
        h, w, _ = frame_shape
        masks = {}
        for region_name, region_data in regions.items():
            points = region_data.get('points', [])
            shape_type = region_data.get('shape_type', 'polygon')

            mask = np.zeros((h, w), dtype=np.uint8)

            if shape_type == 'polygon' and len(points) > 2:
                pts = np.array(points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

            elif shape_type == 'rectangle' and len(points) == 4:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[2][0]), int(points[2][1])
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

            elif shape_type == 'line' and len(points) == 2:
                x1, y1 = int(points[0][0]), int(points[0][1])
                x2, y2 = int(points[1][0]), int(points[1][1])
                cv2.line(mask, (x1, y1), (x2, y2), 255, 2)  # use thickness

            masks[region_name] = mask
        return masks

    def onDisplay(self, region, frame):
        """Display region names on frame"""
        if not region:  # More pythonic than len(region) == 0
            return

        # Display up to the first few regions with proper spacing
        y_offset = 30  # Starting Y position
        line_height = 50  # Space between lines

        # Limit to 5 regions to avoid overcrowding
        for i, reg in enumerate(region[:5]):
            if 'name' in reg:
                y_pos = y_offset + (i * line_height)
                cv2.putText(frame, reg['name'], (10, y_pos),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255))


def _is_connection_alive(source):
    """Check if network connection to source is alive"""
    hostname = urlparse(source).hostname
    param = "-n" if platform.system().lower() == "windows" else "-c"
    command = ["ping", param, "1", hostname]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return 'unreachable' not in result.stdout
    except subprocess.TimeoutExpired:
        return False

async def sendRegularFrames(source, request):
    # Blocking capture/encode work is pushed to threads so the event loop
    # (and every other stream/request) keeps running.
    loop = asyncio.get_running_loop()
    if not await loop.run_in_executor(None, _is_connection_alive, source):
        logging.warning("[Camera Connection not available")
        return
    fresh = FreshestFrame(source)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
    try:
        while fresh.is_alive():
            if await request.is_disconnected():
                logging.info("Client disconnected, releasing camera.")
                break
            _, frame = await loop.run_in_executor(None, fresh.read)
            if frame is None:
                await asyncio.sleep(0.005)
                continue

            _, jpeg = await loop.run_in_executor(
                None, lambda f=frame: cv2.imencode(".jpg", f, encode_params))

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )
    finally:
        fresh.release()

_crop_face_handler = None
_crop_face_lock = threading.Lock()

def _get_crop_face_handler(device: str = None):
    """Get or create cached FaceAnalysis handler for image_crop.

    Providers follow the requested device (or CUDA availability) instead
    of unconditionally demanding CUDA, which broke CPU-only machines.
    """
    global _crop_face_handler
    if _crop_face_handler is None:
        with _crop_face_lock:
            if _crop_face_handler is None:
                use_cuda = (
                    device == 'cuda' or
                    (device is None and torch.cuda.is_available()))
                _crop_face_handler = FaceAnalysis(
                    'antelopev2',
                    providers=(['CUDAExecutionProvider',
                                'CPUExecutionProvider'] if use_cuda
                               else ['CPUExecutionProvider']),
                    root='.'
                )
                prepare_face_handler(_crop_face_handler)
    return _crop_face_handler

def image_crop(filepath, isSearch):
    """Crop face from image with padding"""
    if isSearch:
        frame = cv2.imread(filepath)
        _, img_encoded = cv2.imencode(".jpg", frame)
        return img_encoded
    try:
        face_handler = _get_crop_face_handler()

        frame = cv2.imread(filepath)
        if frame is None:
            raise ValueError(f"Could not load image: {filepath}")

        with _crop_face_lock:
            faces = face_handler.get(frame)
        if not faces:
            raise ValueError("No faces detected in image")

        facebox = select_primary_face(faces).bbox
        x1, y1, x2, y2 = map(int, facebox)

        height_f, width_f = frame.shape[:2]
        x1 = max(x1 - FACE_CROP_PADDING, 0)
        y1 = max(y1 - FACE_CROP_PADDING, 0)
        x2 = min(x2 + FACE_CROP_PADDING, width_f)
        y2 = min(y2 + FACE_CROP_PADDING, height_f)

        cropped_frame = frame[y1:y2, x1:x2]
        _, img_encoded = cv2.imencode(".jpg", cropped_frame)
        return img_encoded

    except Exception as e:
        logging.error(f"Error in image_crop: {e}")
        return None


def takeFrame(rtspurl, filename):
    """Capture a single frame from an RTSP stream.

    Returns (full_frame_encoded, file_location) on success, where
    full_frame_encoded is the JPEG-encoded full frame (not cropped).
    Returns None on failure.
    """
   
    cap = cv2.VideoCapture()
    # Best-effort timeouts so a dead RTSP host can't hang the request
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    cap.open(rtspurl, cv2.CAP_FFMPEG)
    try:
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        cv2.imwrite(filename, frame)
        # Return the full frame encoded (for face selection UI)
        _, full_encoded = cv2.imencode(".jpg", frame)
        return full_encoded
    finally:
        cap.release()
    



if __name__ == "__main__":
    result = image_crop(r'dbimage\aref\image.png')
    if result is not None:
        logging.info("Image cropped successfully")
    else:
        logging.error("Failed to crop image")
