
from asyncio import Queue
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



class CCtvMonitor:
    def __init__(self):
        self.process = None
        self.start()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.frps = 5 if self.device == 'cuda' else 25
        self.fileEx = 'onnx' if self.checkOnnx() else 'pt'
        self.MODEL_PATH = os.getenv(
            "MODEL_PATH", f"models/yolov8n.{self.fileEx}")
        self.TARGET_FPS = 30
        self.FRAME_DELAY = 1.0 / self.TARGET_FPS
        self.RETRY_LIMIT = 5
        self.RETRY_DELAY = 3
        self.ip_relay, self.ip_port, self.relayN1, self.relayN2 = '', '', '', ''
        self.score, self.padding, self.quality, self.hscore, self.simscore, self.port, self.isRegionMode, self.isRelay, self.iou = self.loadConfig()

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
                'antelopev2',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
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
        """Pre-build flat numpy matrix for fast batch cosine similarity"""
        all_embeddings = []
        self._embedding_labels = []  # parallel list of (name, age, gender, role)

        for name, person_data in self.known_names.items():
            age = person_data.get('age', 'None')
            gender = person_data.get('gender', 'None')
            role = person_data.get('role', '')
            for emb in person_data.get('embeddings', []):
                all_embeddings.append(emb)
                self._embedding_labels.append((name, age, gender, role))

        if all_embeddings:
            self._embedding_matrix = np.array(all_embeddings, dtype=np.float32)
            # Normalize all rows once
            norms = np.linalg.norm(self._embedding_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self._embedding_matrix = self._embedding_matrix / norms
        else:
            self._embedding_matrix = np.empty((0, 512), dtype=np.float32)
            self._embedding_labels = []

        logging.info(f"Embedding index built: {len(self._embedding_labels)} vectors")

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
        
        if self.config.isRegionMode:
            self.background_subtractor = cv2.createBackgroundSubtractorMOG2()
            self.k = []

    def start(self):
        self.running = True
        self.stop_event.clear()

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

            
        
    def sendFrames(self):
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
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
                    self.frame_queue.put_nowait((f'/rt{camera_idx}', counter, regions))
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

                # Run YOLO detection
                results = self.config.model.track(
                    masked_frame if self.config.isRegionMode else processed_frame,
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
                        if self.config.isRegionMode:
                            region_name = self.get_detection_region(
                                (x1, y1, x2, y2), region_masks)
                            if region_name and region_name in regions:
                                region_data = regions[region_name]
                                if region_data not in current_regions:
                                    current_regions.append(region_data)
                        else:
                            region_data = None

                        # Get tracking ID

                        track_id = int(res.boxes.id[i])

                        # Crop human region
                        human_crop = masked_frame[y1:y2,
                                                  x1:x2] if self.config.isRegionMode else processed_frame[y1:y2, x1:x2]
                        if human_crop.size == 0:
                            continue

                        # Draw bounding box
                        cv2.rectangle(processed_frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)

                        # Queue for recognition every frps frames
                        # if counter % self.frps == 0:
                        if track_id not in self.processed_tracks:
                            try:
                                self.recognition_queue.put(
                                    (         path,
                                    track_id,
                                    human_crop.copy(),
                                    region_data))
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
                                    'role': ''
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

            except Exception as e:
                logging.error(f"Error processing frame: {e}")

    def recognition_worker(self):
        """Background worker for face recognition with batch queue draining"""
        logging.info("Recognition worker started.")

        while not self.stop_event.is_set():
            try:
                item = self.recognition_queue.get(timeout=0.05)
                if item is None:
                    break

                # Drain queue, keep only latest per track_id
                latest_items = {item[1]: item}
                while not self.recognition_queue.empty():
                    try:
                        next_item = self.recognition_queue.get_nowait()
                        if next_item is None:
                            break
                        latest_items[next_item[1]] = next_item
                    except queue.Empty:
                        break

                for path, track_id, face_img, region_data in latest_items.values():
                    with self.face_info_lock:
                        if (track_id in self.face_info and
                                time.time() - self.face_info[track_id]['last_update'] < RECOGNITION_UPDATE_INTERVAL):
                            continue

                    faces = self.config.face_handler.get(face_img)

                    if faces:
                        face = faces[0]
                        gender = 'female' if face.gender == 0 else 'male'
                        age = face.age
                        det_score = float(face.det_score)
                        if det_score > self.config.score:
                            name, sim, gender, age, role = self.recognize_face(
                                face.embedding, gender, age
                            )

                            x1, y1, x2, y2 = map(int, face.bbox)

                            self.update_face_info(
                                track_id, name, sim, gender, age, role, (
                                    x1, y1, x2, y2)
                            )
                            self.embedding_cache[track_id] = face.embedding

                            height_f, width_f = face_img.shape[:2]
                            padding = self.config.padding
                            fx1_padded = max(x1 - padding, 0)
                            fy1_padded = max(y1 - padding, 0)
                            fx2_padded = min(x2 + padding, width_f)
                            fy2_padded = min(y2 + padding, height_f)

                            cropped_face = face_img[fy1_padded:fy2_padded,
                                                    fx1_padded:fx2_padded]

                            try:
                                read_idx = self.capture_read_idx
                                current_full_frame = self.capture_buffer[read_idx]
                                insertToDb(name, current_full_frame.copy() if current_full_frame is not None else None, cropped_face.copy(), face_img.copy(
                                ), det_score, track_id, gender, age, role, path, self.config.quality, region_data, self.config.isRelay, self.config.isRegionMode, self.config.ip_relay, self.config.ip_port, self.config.relayN1, self.config.relayN2)
                                self.processed_tracks.add(track_id)
                            except Exception as e:
                                logging.error(f"Error inserting to DB: {e}")
                        else:
                            self.update_face_info(
                                track_id, "Unknown", 0.0, 'None', 'None', '', None
                            )
                    else:
                        self.update_face_info(
                            track_id, "Unknown", 0.0, 'None', 'None', '', None
                        )

            except queue.Empty:
                continue
        logging.info("Recognition worker stopped.")

    def recognize_face(self, embedding, fgender, fage):
        """Recognize face using batch vectorized cosine similarity"""
        if self.config._embedding_matrix.shape[0] == 0:
            return "unknown", 0.0, fgender, fage, ''

        query = embedding.astype(np.float32)
        query_norm = np.linalg.norm(query)
        if query_norm > 0:
            query = query / query_norm

        sims = self.config._embedding_matrix @ query
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        if best_score >= self.config.simscore:
            name, age, gender, role = self.config._embedding_labels[best_idx]
            return name, best_score, gender, age, role

        return "unknown", best_score, fgender, fage, ''

    def update_face_info(self, track_id, name, score, gender, age, role, bbox=None):
        """Thread-safe update of face information"""
        with self.face_info_lock:
            self.face_info[track_id] = {
                'name': name,
                'bbox': bbox,
                'last_update': time.time(),
                'score': score,
                'gender': gender,
                'age': age,
                'role': role
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
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
        return 'unreachable' not in result.stdout
    except subprocess.TimeoutExpired:
        return False

async def sendRegularFrames(source, request):
    if not _is_connection_alive(source):
        logging.warning("[Camera Connection not available")
        return
    fresh = FreshestFrame(source)
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 70, cv2.IMWRITE_JPEG_OPTIMIZE, 0]
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
            'antelopev2',
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