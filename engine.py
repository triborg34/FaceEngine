
from asyncio import Queue
from dataclasses import dataclass
import gc
import logging
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
from sklearn.metrics.pairwise import cosine_similarity
import torch
from concurrent.futures import ThreadPoolExecutor
from camera import FreshestFrame
from savatoDb import load_embeddings_from_db, insertToDb
from PIL import Image
from torchvision.transforms import transforms
import json


# --- Basic Setup ---
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.DEBUG,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("log.txt", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

cv2.setNumThreads(multiprocessing.cpu_count())

# --- Constants ---
FACE_CROP_PADDING = 40
SIMILARITY_THRESHOLD = 0.7
FACE_DETECTION_CONFIDENCE_THRESHOLD = 0.5
RECOGNITION_UPDATE_INTERVAL = 2  # seconds
JPEG_QUALITY = 85
TRACK_TTL = 30


@dataclass(frozen=True)
class FaceIndex:
    """عکس فوری و تغییرناپذیر از دیتابیس چهره‌ها."""
    matrix: np.ndarray        # (N, 512) نرمال‌شده
    # هر ردیف: (name, age, gender, role, socialnumber)
    labels: tuple
    name_to_idx: dict         # name -> np.array شماره ردیف‌های آن شخص


class CCtvMonitor:
    def __init__(self, device):
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
        self.min_margin = 0.08
        self.ip_relay, self.ip_port, self.relayN1, self.relayN2 = '', '', '', ''
        self.score, self.padding, self.quality, self.hscore, self.simscore, self.port, self.isRegionMode, self.isRelay, self.iou = self.loadConfig()
        # seconds between samples while voting (faster than RECOGNITION_UPDATE_INTERVAL)
        self.voting_interval = 0.15
        self.min_face_width = 60
        self.min_det_score = 0.6
        self.votes_required = 4       # samples needed before committing to an identity
        # Initialize models
        self.model = None
        self.face_handler = None
        self._load_models()
        self.known_names = self.load_db()
        self._build_embedding_index()

        # Threading and process management
        self.embedding_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._shutdown_event = threading.Event()

        # Image Searcher
        self.FOLDER_PATH = "outputs/humancrop"             # folder containing all images
        self.EMBEDDING_FILE = "embeddings.npy"  # file to save/load embeddings
        self.FILENAMES_FILE = "filenames.txt"  # file to save/load filenames
        self.LOCAL_WEIGHTS = "models/resnet50-0676ba61.pth"
        self.IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # regions

        self.loadWebBrowser(self.port)

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
        with open('iou.txt') as file:
            iou = file.readline()
        iou = float(iou)
        uri = 'http://127.0.0.1:8091/api/collections/setting/records'
        response = requests.get(uri)
        data = response.json().get('items')[0]
        if data['isRfid']:
            self.ip_relay, self.ip_port, self.relayN1, self.relayN2 = data['rfidip'].strip(
            ), data['rfidport'], data['rl1'], data['rl2']
        if data['rl1']:
            self.relayN1 = 1
        if data['rl2']:
            self.relayN2 = 2
        return float(data['score']), data['padding'], int(data['quality']), float(data['hscore']), float(data['simscore']), data['port'], data['isregion'], data['isRfid'], iou

    def load_image_searcher_model(self):
        model = resnet50(weights=None)  # don't load default
        # load weights from file
        state_dict = torch.load(self.LOCAL_WEIGHTS, map_location=self.device)
        model.load_state_dict(state_dict)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval().to(self.device)
        return model

    def get_embedding(self, img_path):
        model = self.load_image_searcher_model()
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = model(img_t)
        features = features.view(features.size(0), -1).cpu().numpy().flatten()
        return features / np.linalg.norm(features)

    def precompute_embeddings(self, model, folder_path):
        logging.info("Precomputing embeddings for all images in folder...")
        embeddings = []
        filenames = []
        for fname in os.listdir(folder_path):
            if not fname.lower().endswith(self.IMG_EXTENSIONS):
                continue
            fpath = os.path.join(folder_path, fname)
            emb = self.get_embedding(fpath)
            embeddings.append(emb)
            filenames.append(fname)
            logging.info(f"Processed {fname}")
        embeddings = np.array(embeddings)
        np.save(self.EMBEDDING_FILE, embeddings)
        with open(self.FILENAMES_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(filenames))
        logging.info(
            f"Saved embeddings to {self.EMBEDDING_FILE} and filenames to {self.FILENAMES_FILE}")
        return embeddings, filenames

    def load_embeddings(self):
        embeddings = np.load(self.EMBEDDING_FILE)
        with open(self.FILENAMES_FILE, "r", encoding='utf-8') as f:
            filenames = f.read().splitlines()
        logging.info(f"Loaded {len(filenames)} embeddings from disk")
        return embeddings, filenames

    def find_similar_images(self, query_embedding, embeddings, filenames, top_k=10):
        sims = cosine_similarity([query_embedding], embeddings)[0]
        if sims[0] > SIMILARITY_THRESHOLD:
            sorted_indices = np.argsort(sims)[::-1]
            results = [(filenames[i], sims[i]) for i in sorted_indices[:top_k]]
            return results
        return []

    def _load_models(self):
        """Load YOLO and face recognition models"""
        try:
            logging.info(f"Loading models...")

            # Load face handler
            self.face_handler = FaceAnalysis(
                'buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else [
                    'CPUExecutionProvider'],
                root='.'
            )
            self.face_handler.prepare(ctx_id=0)

            # Load YOLO model
            if self.device == 'cpu' and self.checkOpenVino():
                logging.info('Loadin openvino')
                self.model = YOLO('models/yolov8n_openvino_model',
                                  task='detect', verbose=False)
            else:
                logging.info('Loadin onnx/pt')
                self.model = YOLO(
                    self.MODEL_PATH, task='detect', verbose=False)
                if self.fileEx != 'onnx':
                    self.model.eval()

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
        all_embeddings = []
        labels = []
        name_to_idx = {}

        for name, person_data in self.known_names.items():
            age = person_data.get('age', 'None')
            gender = person_data.get('gender', 'None')
            role = person_data.get('role', '')
            socialnumber = person_data.get('socialnumber', '')
            for emb in person_data.get('embeddings', []):
                idx = len(all_embeddings)
                all_embeddings.append(emb)
                labels.append((name, age, gender, role, socialnumber))
                name_to_idx.setdefault(name, []).append(idx)

        if all_embeddings:
            m = np.array(all_embeddings, dtype=np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0] = 1
            m = m / norms
        else:
            m = np.empty((0, 512), dtype=np.float32)

        new_index = FaceIndex(
            m, tuple(labels), {k: np.asarray(v) for k, v in name_to_idx.items()})

        # فقط یک انتساب => اتمیک. بقیه‌ی ترد‌ها یا نسخه‌ی قدیمی را می‌بینند یا جدید را،
        # هیچ‌وقت نسخه‌ی نیمه‌کاره را.
        self.index = new_index

        logging.info(
            f"Embedding index built: {len(labels)} vectors "
            f"across {len(name_to_idx)} identities")

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
    def __init__(self, source, config: CCtvMonitor, camera_id):
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

        # ========== CHANGE 1: LOCK-FREE BUFFERS ==========
        self.capture_buffer = [None, None]  # For raw frames from camera
        self.display_buffer = [None, None]  # For processed frames
        self.capture_write_idx = 0
        self.capture_read_idx = 0
        self.display_write_idx = 0
        self.display_read_idx = 0

        self.capture_version = 0
        self.display_version = 0

        # ========== CHANGE 2: OPTIMIZED QUEUES ==========
        # Smaller queues, faster operations
        self.frame_queue = queue.Queue(maxsize=2)  # Was 10
        self.recognition_queue = queue.Queue(maxsize=3)  # Was 10

        # ---------- DATA ----------
        # Remove: self.result_frame, self.result_lock
        self.processed_tracks = set()
        self.face_info = {}
        self.face_info_lock = threading.Lock()
        self.embedding_cache = {}
        self.last_queued_at = {}       # track_id -> last time it was queued for recognition
        self.track_votes = {}          # track_id -> list of pending recognition samples
        self.track_decided = set()     # track_ids that have a locked-in identity
        self.track_last_seen = {}
        self.db_queue = queue.Queue(maxsize=100)
        self.db_thread = None

        if self.config.isRegionMode:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
            self.k = []

    def start(self):
        self.running = True
        self.stop_event.clear()

        self.capture_thread = threading.Thread(
            target=self.generate_frames, args=[
                self.camera_id, self.source], daemon=True
        )
        self.process_thread = threading.Thread(
            target=self.process_frame, daemon=True
        )
        self.recognition_thread = threading.Thread(
            target=self.recognition_worker,
            daemon=True,
        )
        self.db_thread = threading.Thread(target=self._db_writer, daemon=True)


        self.capture_thread.start()
        self.process_thread.start()
        self.recognition_thread.start()
        self.db_thread.start()

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

    def sendFrames(self):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY,
                         70, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
        last_version = -1
        while self.running:
            current_version = self.display_version
            if current_version == last_version:
                time.sleep(0.003)
                continue
            last_version = current_version
            read_idx = self.display_read_idx
            frame = self.display_buffer[read_idx]

            if frame is None:
                time.sleep(0.003)
                continue

            _, jpeg = cv2.imencode(".jpg", frame, encode_params)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

    def generate_frames(self, camera_idx, source):
        """Generate frames from a specific camera feed"""
        if not self.is_connection_alive(source):
            logging.warning(f"[Camera {camera_idx}] Connection not available")
            return

        counter = 0
        if self.config.isRegionMode:
            regions = self.load_regions(soruce=source)
            if regions == None:
                regions = {"r2": {
                    "id": "1345",
                    "name": "r2",
                    "description": "",
                    "points": [
                        [0.0, 0.0],          # top-left
                        [999.0, 0.0],  # top-right
                        [999.0, 999.0],  # bottom-right
                        [0.0, 999.0],       # bottom-left
                        [0.0, 0.0]
                    ],
                    "shape_type": "polygon",
                    "color": "red",
                    "created": "2025-08-05T11:46:12.379819",

                    "ip": urlparse(source).hostname
                }, }

            if not hasattr(self, 'k'):
                self.k = []
        else:
            regions = None

        fresh = FreshestFrame(source)

        try:
            while self.running and not self.stop_event.is_set():

                success, frame = fresh.read()
                counter += 1

                # if counter%750  ==0:
                #     print("CLEARING TRACKS")
                #     self.processed_tracks.clear()

                if frame is None:
                    continue
                write_idx = self.capture_write_idx
                self.capture_buffer[write_idx] = frame

                # Swap buffers atomically
                self.capture_read_idx = write_idx
                self.capture_write_idx = 1 - write_idx
                self.capture_version += 1

                # Process frame
                try:
                    self.frame_queue.put_nowait(
                        (f'/rt{camera_idx}', counter, regions))
                except queue.Full:
                    pass

        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")
        finally:
            logging.info("Releasing camera resources")
            fresh.release()

    def _db_writer(self):
        """رویدادها را از صف می‌گیرد و insertToDb را جدا از ترد تشخیص اجرا می‌کند.
        بعد از stop_event هم صف را خالی می‌کند تا آخرین رویدادها گم نشوند."""
        while True:
            try:
                args = self.db_queue.get(timeout=0.2)
            except queue.Empty:
                if self.stop_event.is_set():
                    break
                continue
            try:
                insertToDb(*args)
            except Exception as e:
                logging.error(f"Error inserting to DB: {e}")

    def is_connection_alive(self, source):
        """Check if network connection to source is alive"""
        return _is_connection_alive(source)

    def process_frame(self):
        last_capture_version = -1
        last_cleanup = time.time()
        """Process a single frame for object detection and face recognition"""
        while self.running:
            try:
                item = self.frame_queue.get(timeout=0.05)
            except queue.Empty:
                if not self.running:
                    break
                continue

            if item is None:
                logging.info("process_frame shutdown signal received")
                break
            path, counter, regions = item
            current_capture_version = self.capture_version

            # Skip if same frame
            if current_capture_version == last_capture_version:
                continue

            last_capture_version = current_capture_version

            # Read from stable read buffer
            read_idx = self.capture_read_idx
            frame = self.capture_buffer[read_idx]
            if frame is None or frame.size == 0:
                continue

            try:

                start_time = time.time()
                processed_frame = frame.copy()

                if self.config.isRegionMode:
                    region_masks = self.generate_region_masks(
                        processed_frame.shape, regions)
                    combined_mask = np.zeros(
                        processed_frame.shape[:2], dtype=np.uint8)
                    for mask in region_masks.values():
                        combined_mask = cv2.bitwise_or(combined_mask, mask)
                    masked_frame = cv2.bitwise_and(
                        processed_frame, processed_frame, mask=combined_mask)
                    self.k.clear()
                    current_regions = []
                source = masked_frame if self.config.isRegionMode else frame
                # Run YOLO detection
                results = self.config.model.track(
                    source,
                    classes=[0],  # Person class
                    iou=self.config.iou,
                    tracker="bytetrack.yaml",
                    persist=True,
                    device=self.config.device,
                    conf=self.config.hscore,
                )

                for res in results:
                    if res.boxes.id is None:
                        continue
                    for i in range(len(res.boxes.xyxy)):
                        x1, y1, x2, y2 = res.boxes.xyxy[i].int().tolist()
                        H, W = source.shape[:2]
                        x1, y1 = max(x1, 0), max(y1, 0)
                        x2, y2 = min(x2, W), min(y2, H)
                        if x2 <= x1 or y2 <= y1:
                            continue

                        # FIX #1: region_data must always be defined before use,
                        # even when isRegionMode is True but no region matched.
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
                        self.track_last_seen[track_id] = time.time()

                        # Crop human region
                        human_crop = source[y1:y2, x1:x2]
                        if human_crop.size == 0:
                            continue

                        # Draw bounding box
                        cv2.rectangle(processed_frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)

                        # FIX #2 + VOTING: time-based re-queue gate. While a track
                        # hasn't reached a decision yet, poll fast (VOTING_INTERVAL)
                        # to gather enough samples quickly. Once decided, fall back
                        # to the slow periodic interval (just to refresh bbox / catch
                        # drift), instead of never re-queuing or re-queuing every frame.
                        now = time.time()
                        last_queued = self.last_queued_at.get(track_id, 0)
                        interval = (RECOGNITION_UPDATE_INTERVAL
                                    if track_id in self.track_decided
                                    else self.config.voting_interval)
                        if now - last_queued >= interval:
                            self.last_queued_at[track_id] = now
                            try:
                                self.recognition_queue.put_nowait(
                                    (path, track_id, human_crop.copy(), region_data))
                            except queue.Full:
                                pass

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
                            cv2.rectangle(
                                processed_frame,
                                (x1 + fx1, y1 + fy1),
                                (x1 + fx2, y1 + fy2),
                                (0, 0, 255), 2
                            )
                            cv2.putText(
                                processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                            )
                        else:
                            cv2.putText(
                                processed_frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                            )

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
                write_idx = self.display_write_idx
                self.display_buffer[write_idx] = display_frame

                # Swap buffers atomically
                self.display_read_idx = write_idx
                self.display_write_idx = 1 - write_idx
                self.display_version += 1
                if time.time() - last_cleanup > 10:
                    last_cleanup = time.time()
                    self.cleanup_tracks(last_cleanup)

            except Exception as e:
                logging.error(f"Error processing frame: {e}")

    def recognition_worker(self):
        logging.info("Recognition worker started.")
    
        while not self.stop_event.is_set():
            try:
                item = self.recognition_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if item is None:
                break
    
            # فقط آخرین آیتم هر track نگه داشته می‌شود
            latest_items = {item[1]: item}
            shutdown = False
            while True:
                try:
                    nxt = self.recognition_queue.get_nowait()
                except queue.Empty:
                    break
                if nxt is None:
                    shutdown = True
                    break
                latest_items[nxt[1]] = nxt
    
            for path, track_id, face_img, region_data in latest_items.values():
                # track ممکن است در فاصله‌ی صف شدن توسط cleanup پاک شده باشد
                if track_id not in self.track_last_seen:
                    logging.warning(f"track {track_id} skipped, known: {list(self.track_last_seen)}")
                    continue
                try:
                    self._handle_track(path, track_id, face_img, region_data)
                except Exception as e:
                    # خطا در یک track نباید کل ترد تشخیص را بکشد
                    logging.exception(f"recognition error on track {track_id}: {e}")
    
            if shutdown:
                break
    
        logging.info("Recognition worker stopped.")
    
    def cleanup_tracks(self, now):
            dead = [t for t, ts in list(self.track_last_seen.items())
                    if now - ts > TRACK_TTL]
            for t in dead:
                self.track_last_seen.pop(t, None)
                self.last_queued_at.pop(t, None)
                self.track_votes.pop(t, None)
                self.track_decided.discard(t)
                self.processed_tracks.discard(t)
                self.embedding_cache.pop(t, None)
                with self.face_info_lock:
                    self.face_info.pop(t, None)
            if dead:
                logging.debug(f"cleanup_tracks: removed {len(dead)} stale tracks")

    def recognize_face(self, embedding, fgender, fage, top_k=3):
            idx = self.config.index          # فقط یک‌بار خوانده می‌شود
            if idx.matrix.shape[0] == 0:
                return "unknown", 0.0, fgender, fage, '', ''

            query = embedding.astype(np.float32)
            qn = np.linalg.norm(query)
            if qn > 0:
                query = query / qn

            sims = idx.matrix @ query

            name_scores = {}
            for name, rows in idx.name_to_idx.items():
                name_sims = sims[rows]
                k = min(top_k, len(name_sims))
                name_scores[name] = float(
                    np.mean(np.partition(name_sims, -k)[-k:]))

            if not name_scores:
                return "unknown", 0.0, fgender, fage, '', ''

            ranked = sorted(name_scores.items(),
                            key=lambda kv: kv[1], reverse=True)
            best_name, best_score = ranked[0]
            logging.info("match top: " + ", ".join(f"{n}={s:.3f}" for n, s in ranked[:3]))
            second_name, second_score = ranked[1] if len(
                ranked) > 1 else (None, None)

            if best_score < self.config.simscore:
                return "unknown", max(best_score, 0.0), fgender, fage, '', ''

            min_margin = getattr(self.config, 'min_margin', 0.08)
            if second_score is not None and (best_score - second_score) < min_margin:
                logging.info(
                    f"Ambiguous match: {best_name}={best_score:.3f} vs "
                    f"{second_name}={second_score:.3f}")
                return "unknown", best_score, fgender, fage, '', ''

            row = idx.name_to_idx[best_name][0]
            _, age, gender, role, socialnumber = idx.labels[row]
            return best_name, best_score, gender, age, role, socialnumber

    def _resolve_votes(self, votes):
            """
            Resolve accumulated (name, sim, gender, age, role, socialnumber) samples
            for a track into a single identity decision.

            Weighted majority: each candidate name's "score" is the sum of its
            similarity scores across samples (not just a raw count), so one
            high-confidence match outweighs two marginal ones. The winner must
            also account for a strict majority of samples, or we fall back to
            "unknown" rather than committing on a split vote.
            """
            tally = {}
            for name, sim, gender, age, role, socialnumber in votes:
                entry = tally.setdefault(
                    name, {'total_sim': 0.0, 'count': 0, 'last': None})
                entry['total_sim'] += sim
                entry['count'] += 1
                entry['last'] = (gender, age, role, socialnumber)

            best_name = max(tally, key=lambda n: tally[n]['total_sim'])
            best = tally[best_name]

            if best_name == "unknown" or best['count'] * 2 <= len(votes):
                # no real majority — don't commit to an identity
                gender, age, role, socialnumber = votes[-1][2], votes[-1][3], votes[-1][4], votes[-1][5]
                return "unknown", 0.0, gender, age, role, socialnumber

            avg_sim = best['total_sim'] / best['count']
            gender, age, role, socialnumber = best['last']
            return best_name, avg_sim, gender, age, role, socialnumber

    def update_face_info(self, track_id, name, score, gender, age, role, socialnumber, bbox=None):
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
                    'socialnumber': socialnumber
                }

    def release_resources(self, role=False):
            # if fresh is not None:
            #     fresh.release()
            if not self.running:
                return

            self.running = False
            logging.info("Camera pipeline stopped")

            # except Exception as e:
            #     logging.error(f"Error releasing camera resources: {e}")

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

                    # Add background for text
                    text = f"{region_name} (ID: {region_data.get('id', 'N/A')})"
                    text_size = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    # cv2.rectangle(overlay, (center_x - text_size[0]//2 - 5, center_y - text_size[1] - 5),
                    #               (center_x + text_size[0]//2 + 5, center_y + 5), (0, 0, 0), -1)
                    # cv2.putText(overlay, text, (center_x - text_size[0]//2, center_y),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

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

    @staticmethod
    def _best_face(faces):
        """کیفیت × عرض چهره؛ به‌جای faces[0]."""
        return max(faces, key=lambda f: float(f.det_score) * (f.bbox[2] - f.bbox[0]))
    
    def _quality_gate_example(self, face, det_score, track_id):
        fw = int(face.bbox[2] - face.bbox[0])       # عرض چهره به پیکسلِ فریم اصلی
        logging.info(f"track {track_id}: face_w={fw}px det={det_score:.2f}")
    
        min_w = getattr(self.config, 'min_face_width', 60)
        min_det = getattr(self.config, 'min_det_score', 0.6)
        if fw < min_w or det_score < min_det:
            # چهره برای امبدینگ قابل‌اعتماد خیلی کوچک/ضعیف است: رأی نده
            self.update_face_info(track_id, "Analyzing...", 0.0, 'None', 'None', '', '', None)
            return False
        return True
    def _handle_track(self, path, track_id, face_img, region_data):
        faces = self.config.face_handler.get(face_img)
        faces = [f for f in faces if float(f.det_score) > self.config.score]
    
        # --- track قبلاً تصمیم گرفته: فقط bbox را برای نمایش تازه کن ---
        if track_id in self.track_decided:
            if faces:
                face = self._best_face(faces)
                x1, y1, x2, y2 = map(int, face.bbox)
                with self.face_info_lock:
                    existing = self.face_info.get(track_id)
                if existing:
                    self.update_face_info(
                        track_id, existing['name'], existing['score'],
                        existing['gender'], existing['age'],
                        existing['role'], existing['socialnumber'],
                        (x1, y1, x2, y2))
            return
    
        if not faces:
            self.update_face_info(track_id, "Analyzing...", 0.0, 'None', 'None', '', '', None)
            return
    
        face = self._best_face(faces)
        gender = 'female' if face.gender == 0 else 'male'
        age = face.age
        det_score = float(face.det_score)
        x1, y1, x2, y2 = map(int, face.bbox)
        if not self._quality_gate_example(face, det_score, track_id):
            return
    
        name, sim, gender, age, role, socialnumber = self.recognize_face(
            face.embedding, gender, age)
    
        votes = self.track_votes.setdefault(track_id, [])
        votes.append((name, sim, gender, age, role, socialnumber))
        self.update_face_info(track_id, "Analyzing...", sim, gender, age, role,
                            socialnumber, (x1, y1, x2, y2))
    
        if len(votes) < self.config.votes_required:
            return
    
        # --- تصمیم نهایی ---
        (final_name, final_sim, final_gender, final_age,
        final_role, final_social) = self._resolve_votes(self.track_votes.pop(track_id))
        self.track_decided.add(track_id)
        self.update_face_info(track_id, final_name, final_sim, final_gender, final_age,
                            final_role, final_social, (x1, y1, x2, y2))
    
        if track_id in self.processed_tracks:
            return
    
        Hf, Wf = face_img.shape[:2]
        pad = self.config.padding
        cropped_face = face_img[max(y1 - pad, 0):min(y2 + pad, Hf),
                                max(x1 - pad, 0):min(x2 + pad, Wf)]
        full = self.capture_buffer[self.capture_read_idx]
    
        args = (final_name,
                full.copy() if full is not None else None,
                cropped_face.copy(), face_img.copy(),
                det_score, track_id, final_gender, final_age, final_role, final_social,
                path, self.config.quality, region_data,
                self.config.isRelay, self.config.isRegionMode,
                self.config.ip_relay, self.config.ip_port,
                self.config.relayN1, self.config.relayN2)
        try:
            self.db_queue.put_nowait(args)      # بدون بلاک
            self.processed_tracks.add(track_id)
        except queue.Full:
            logging.warning("db_queue full, dropping event")


def image_searcher(file_path):
    """Load and encode image for searching"""
    try:
        frame = cv2.imread(file_path)
        if frame is None:
            raise ValueError(f"Could not load image: {file_path}")
        _, img_encoded = cv2.imencode(".jpg", frame)
        return img_encoded
    except Exception as e:
        logging.error(f"Error in image_searcher: {e}")
        return None


def _is_connection_alive(source):
    """Check if network connection to source is alive"""
    hostname = urlparse(source).hostname
    param = "-n" if platform.system().lower() == "windows" else "-c"
    command = ["ping", param, "1", hostname]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=10)
        return 'unreachable' not in result.stdout
    except subprocess.TimeoutExpired:
        return False


async def sendRegularFrames(source, request):
    if not _is_connection_alive(source):
        logging.warning("[Camera Connection not available")
        return
    fresh = FreshestFrame(source)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY,
                     70, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
    while fresh.is_alive():
        if await request.is_disconnected():
            logging.info("Client disconnected, releasing camera.")
            break
        success, frame = fresh.read()
        if frame is None:
            time.sleep(0.005)
            continue

        _, jpeg = cv2.imencode(".jpg", frame, encode_params)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + jpeg.tobytes()
            + b"\r\n"
        )
    fresh.release()

_crop_face_handler = None


def _get_crop_face_handler():
    """Get or create cached FaceAnalysis handler for image_crop"""
    global _crop_face_handler
    if _crop_face_handler is None:
        _crop_face_handler = FaceAnalysis(
            'buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            root='.'
        )
        _crop_face_handler.prepare(ctx_id=0)
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

        faces = face_handler.get(frame)
        if not faces:
            raise ValueError("No faces detected in image")

        facebox = faces[0].bbox
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
    print(rtspurl)
    try:
        cap = cv2.VideoCapture(rtspurl)
    except Exception as e:
        return

    ret, frame = cap.read()
    if frame is None:
        return
    cv2.imwrite(f'{filename}', frame)
    face_handler = _get_crop_face_handler()
    faces = face_handler.get(frame)
    if not faces:
        raise ValueError("No faces detected in image")
    facebox = faces[0].bbox
    x1, y1, x2, y2 = map(int, facebox)

    height_f, width_f = frame.shape[:2]
    x1 = max(x1 - FACE_CROP_PADDING, 0)
    y1 = max(y1 - FACE_CROP_PADDING, 0)
    x2 = min(x2 + FACE_CROP_PADDING, width_f)
    y2 = min(y2 + FACE_CROP_PADDING, height_f)

    cropped_frame = frame[y1:y2, x1:x2]
    _, img_encoded = cv2.imencode(".jpg", cropped_frame)
    return img_encoded


if __name__ == "__main__":
    result = image_crop(r'dbimage\aref\image.png')
    if result is not None:
        logging.info("Image cropped successfully")
    else:
        logging.error("Failed to crop image")


'''

## **What Changed**

1. **Separate buffers for capture and display:**
   - `capture_buffer[2]` - Raw frames from camera
   - `display_buffer[2]` - Processed frames with detections

2. **Proper read/write index separation:**
   - Each buffer has its own `write_idx` and `read_idx`
   - Writer updates write buffer, then swaps indices
   - Reader always reads from stable read buffer

3. **Version counters:**
   - Detect when new frames are available
   - Prevent processing same frame multiple times

## **How It Works**
```
Camera Thread:
  [Capture Frame] → write to capture_buffer[write_idx]
                 → swap: read_idx = write_idx, write_idx = 1-write_idx
                 → increment capture_version

Process Thread:
  Read from capture_buffer[read_idx] ← STABLE, won't change mid-read
  [Process Frame] → write to display_buffer[write_idx]
                  → swap: read_idx = write_idx, write_idx = 1-write_idx
                  → increment display_version

Send Thread:
  Read from display_buffer[read_idx] ← STABLE, won't change mid-read
  [Encode & Send]
  
'''
