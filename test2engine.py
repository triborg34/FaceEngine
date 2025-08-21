import asyncio
import gc
import io
import logging
import multiprocessing
import os
import platform
import queue
import subprocess
import sys
import time
import threading
from torchvision.models import resnet50
from urllib.parse import urlparse
import cv2
from fastapi import Request
import numpy as np
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import torch
from concurrent.futures import ThreadPoolExecutor
from camera import FreshestFrame
from savatoDb import load_embeddings_from_db, insertToDb
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms

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


class CCtvMonitor:
    def __init__(self):
        # Set device and optimize CUDA settings
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if self.device == 'cuda':
            # Optimize CUDA performance
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            self.frps = 5
        else:
            self.frps = 25
        
        self.MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")
        self.TARGET_FPS = 30
        self.FRAME_DELAY = 1.0 / self.TARGET_FPS
        self.RETRY_LIMIT = 5
        self.RETRY_DELAY = 3

        # Set thread counts and optimize CPU usage
        cpu_count = multiprocessing.cpu_count()
        torch.set_num_threads(cpu_count)
        cv2.setNumThreads(cpu_count)
        
        # Enable OpenCV optimizations
        cv2.ocl.setUseOpenCL(True)  # Enable OpenCL acceleration if available
        
        # Initialize model cache
        self.model = None
        self.face_handler = None
        self.resnet_model = None
        self._load_models()

        # Load database
        self.known_names = self.load_db()
        
        # Image Searcher
        self.FOLDER_PATH = "outputs/humancrop"             # folder containing all images
        self.EMBEDDING_FILE = "embeddings.npy"  # file to save/load embeddings
        self.FILENAMES_FILE = "filenames.txt"  # file to save/load filenames
        self.LOCAL_WEIGHTS = 'models/resnet50-0676ba61.pth'
     
        if not os.path.exists(self.LOCAL_WEIGHTS):
            logging.error(f"ResNet50 weights not found at {self.LOCAL_WEIGHTS}")
            raise FileNotFoundError(f"Required model weights not found at {self.LOCAL_WEIGHTS}")
        self.IMG_EXTENSIONS = (".jpg", ".jpeg", ".png")
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Threading and process management
        self.process = None
        self.lock = threading.Lock()
        self.recognition_queue = queue.Queue()
        self.face_info = {}
        self.face_info_lock = threading.Lock()
        self.embedding_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=10)
        self._shutdown_event = threading.Event()

        

    def load_image_searcher_model(self):
        # Only load once, reuse
        print(self.LOCAL_WEIGHTS)
        if self.resnet_model is not None:
            return self.resnet_model
        model = resnet50(weights=None)
        state_dict = torch.load(self.LOCAL_WEIGHTS, map_location=self.device)
        model.load_state_dict(state_dict)
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval().to(self.device)
        self.resnet_model = model
        return self.resnet_model

    def get_embedding(self, img_path):
        model = self.load_image_searcher_model()
        img = Image.open(img_path).convert("RGB")
        img_t = self.transform(img).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.no_grad():
            features = model(img_t)
        features = features.view(features.size(0), -1).cpu().numpy().flatten()
        return features / np.linalg.norm(features)

    def precompute_embeddings(self, model, folder_path):
        print("Precomputing embeddings for all images in folder...")
        embeddings = []
        filenames = []
        img_paths = [os.path.join(folder_path, fname) for fname in os.listdir(folder_path) if fname.lower().endswith(self.IMG_EXTENSIONS)]
        # Batch processing for speed (if many images)
        for fpath in img_paths:
            emb = self.get_embedding(fpath)
            embeddings.append(emb)
            filenames.append(os.path.basename(fpath))
            print(f"Processed {os.path.basename(fpath)}")
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
        print(f"Loaded {len(filenames)} embeddings from disk")
        return embeddings, filenames

    def find_similar_images(self, query_embedding, embeddings, filenames, top_k=10):
        # Convert to torch tensors for faster computation
        query_tensor = torch.tensor(query_embedding).to(self.device)
        embeddings_tensor = torch.tensor(embeddings).to(self.device)
        
        # Compute similarities using optimized GPU/CPU operations
        with torch.no_grad():
            sims = torch.nn.functional.cosine_similarity(
                query_tensor.unsqueeze(0), 
                embeddings_tensor
            )
            
            # Get top-k results efficiently
            top_sims, indices = torch.topk(sims, k=min(top_k, len(filenames)))
            
        # Convert results back to CPU for processing
        results = [(filenames[i], s.item()) for i, s in zip(indices.cpu().numpy(), top_sims.cpu().numpy())]
        return results

    def _load_models(self):
        """Load YOLO and face recognition models"""
        try:
            logging.info("Loading models...")

            # Load face handler
            self.face_handler = FaceAnalysis(
                'buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
                root='.'
            )
            # Use GPU if available, else CPU
            ctx_id = 0 if self.device == 'cuda' else -1
            self.face_handler.prepare(ctx_id=ctx_id)

            # Load YOLO model
            self.model = YOLO(self.MODEL_PATH, verbose=False)
            self.model.eval()

            # Load ResNet50 for image searcher (done in load_image_searcher_model)
            # self.load_image_searcher_model()
            print(self.frps)
            logging.info('Models loaded successfully.')

        except Exception as e:
            logging.error(f"Failed to load models: {e}")
            raise

    def load_db(self):
        """Load known faces from database"""
        try:
            # process = subprocess.Popen(
            #         #     ["pocketbase", "serve", "--http=0.0.0.0:8091"], creationflags=subprocess.CREATE_NO_WINDOW,)
            #         # logging.info(f"PocketBase stater {process.pid}")
            known_names = load_embeddings_from_db()
            logging.info(
                f"Loaded {len(known_names)} known faces from database")
            return known_names
        except Exception as e:
            logging.error(f"Failed to load database: {e}")
            return {}

    def release_resources(self, fresh: FreshestFrame, cap: cv2.VideoCapture,role:bool):
        """Properly release camera resources"""
        try:
            if fresh:
                fresh.release()
            if cap:
                cap.release()
            # Signal recognition worker to stop
            if not role:
                self.recognition_queue.put(None)
        except Exception as e:
            logging.error(f"Error releasing camera resources: {e}")

    def graceful_shutdown(self):
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

        # Release resnet model
        self.resnet_model = None

        logging.info("Cleanup complete.")

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

    def recognize_face(self, embedding, fgender, fage):
        """Recognize face using optimized embedding comparison"""
        best_match = 'unknown'
        best_score = 0.0
        best_age = fage
        best_gender = fgender
        best_role = ''

        try:
            # Convert query embedding to tensor for faster computation
            query_emb = torch.tensor(embedding).to(self.device)
            
            # Process all embeddings in batches for better performance
            for name, person_data in self.known_names.items():
                age = person_data['age']
                gender = person_data['gender']
                role = person_data['role']
                embeds = person_data['embeddings']
                
                # Convert embeddings to tensor and move to device
                known_embs = torch.tensor(embeds).to(self.device)
                
                # Compute similarities in one go using matrix operations
                with torch.no_grad():
                    sims = torch.nn.functional.cosine_similarity(
                        query_emb.unsqueeze(0), 
                        known_embs
                    )
                    max_sim, _ = torch.max(sims, dim=0)
                    max_sim = max_sim.item()
                
                if max_sim > best_score:
                    best_score = max_sim
                    best_match = name
                    best_age = age
                    best_gender = gender
                    best_role = role

            # Threshold for recognition
            if best_score >= 0.6:
                return best_match, best_score, best_gender, best_age, best_role
            else:
                return "unknown", best_score, fgender, fage, best_role

        except Exception as e:
            logging.error(f"Error in face recognition: {e}")
            return "unknown", 0.0, fgender, fage, ""

    def recognition_worker(self):
        """Background worker for face recognition"""
        logging.info("Recognition worker started.")

        while not self._shutdown_event.is_set():
            try:
                # Use timeout to allow checking shutdown event
                item = self.recognition_queue.get(timeout=1.0)

                if item is None:
                    break

                track_id, face_img = item

                # Skip if recently updated (performance optimization)
                with self.face_info_lock:
                    if (track_id in self.face_info and
                            time.time() - self.face_info[track_id]['last_update'] < 2):
                        continue

                # Process face
                faces = self.face_handler.get(face_img)

                if faces:
                    face = faces[0]
                    gender = 'female' if face.gender == 0 else 'male'
                    age = face.age

                    name, sim, gender, age, role = self.recognize_face(
                        face.embedding, gender, age
                    )
                    x1, y1, x2, y2 = map(int, face.bbox)

                    self.update_face_info(
                        track_id, name, sim, gender, age, role, (
                            x1, y1, x2, y2)
                    )
                    self.embedding_cache[track_id] = face.embedding
                else:
                    self.update_face_info(
                        track_id, "Unknown", 0.0, 'None', 'None', '', None
                    )

            except queue.Empty:
                continue  # Timeout, check shutdown event
            except Exception as e:
                logging.error(f"Error in recognition worker: {e}")

        logging.info("Recognition worker stopped.")

    def start(self):
        # if not os.path.exists(self.EMBEDDING_FILE) or not os.path.exists(self.FILENAMES_FILE):
        # self.precompute_embeddings(
        #         self.load_image_searcher_model(), self.FOLDER_PATH)
        """Start the recognition worker thread"""
        # recognition_thread = threading.Thread(
        #     target=self.recognition_worker,
        #     daemon=True
        # )
        # recognition_thread.start()
        # return recognition_thread

    async def process_frame(self, frame, path, counter):
        """Process a single frame for object detection and face recognition"""
        try:
            if frame.size == 0:
                return frame

            start_time = time.time()

            # Optimize frame for processing
            # if self.device == 'cuda':
            #     # Use GPU-optimized resize for CUDA
            #     processed_frame = cv2.cuda.resize(cv2.cuda_GpuMat(frame), (640, 480)).download()
            # else:
            #     # Efficient CPU resize with area interpolation
            #     processed_frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_AREA)
            processed_frame=frame.copy()
            # Run YOLO detection with optimized settings
            results = self.model.track(
                processed_frame,
                classes=[0],  # Person class
                tracker="bytetrack.yaml",
                persist=True,
                device=self.device,
                conf=0.5,  # Confidence threshold
                iou=0.45,  # NMS IOU threshold
                agnostic_nms=True  # Class-agnostic NMS
            )

            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0][:4].cpu().tolist())

                    # Get tracking ID
                    if box.id is None:
                        continue
                    track_id = int(box.id[0].cpu().item())

                    # Crop human region
                    human_crop = processed_frame[y1:y2, x1:x2]
                    if human_crop.size == 0:
                        continue

                    # Draw bounding box
                    cv2.rectangle(processed_frame, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)

                    # Queue for recognition every frps frames
                    # if counter % self.frps == 0:
                    self.recognition_queue.put((track_id, human_crop))

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

                    try:
                        score = int(info['score'] *
                                    100) if info['score'] else 0
                    except (TypeError, ValueError):
                        score = 0

                    name = info['name']
                    gender = info['gender']
                    age = info['age']
                    role = info['role']

                    # Draw face bounding box if available
                    if face_bbox:
                        fx1, fy1, fx2, fy2 = face_bbox
                        cv2.rectangle(
                            processed_frame,
                            (x1 + fx1, y1 + fy1),
                            (x1 + fx2, y1 + fy2),
                            (0, 0, 255), 1
                        )
                        cv2.putText(
                            processed_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                        )

                        # Crop face with padding
                        height_f, width_f = human_crop.shape[:2]
                        padding = 40
                        fx1_padded = max(fx1 - padding, 0)
                        fy1_padded = max(fy1 - padding, 0)
                        fx2_padded = min(fx2 + padding, width_f)
                        fy2_padded = min(fy2 + padding, height_f)

                        cropped_face = human_crop[fy1_padded:fy2_padded,
                                                  fx1_padded:fx2_padded]

                        # Insert to database
                        try:
                            await insertToDb(
                                name, processed_frame, cropped_face, human_crop,
                                score, track_id, gender, age, role, path
                            )
                        except Exception as e:
                            logging.error(f"Error inserting to DB: {e}")

                    else:
                        cv2.putText(
                            processed_frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
                        )

            # Calculate and display FPS
            # fps = 1.0 / (time.time() - start_time)
            # cv2.putText(
            #     processed_frame, f"FPS: {fps:.2f}", (10, 25),
            #     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
            # )

            return processed_frame

        except Exception as e:
            logging.error(f"Error processing frame: {e}")
            return frame

    def is_connection_alive(self, source):
        """Check if network connection to source is alive"""
        try:
            url = urlparse(source).hostname
            if not url:
                return True  # Local source or invalid URL

            param = "-n" if platform.system().lower() == "windows" else "-c"
            command = ["ping", param, "1", url]

            result = subprocess.run(
                command, capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0

        except (subprocess.TimeoutExpired, Exception) as e:
            logging.warning(f"Connection check failed: {e}")
            return False

    async def generate_frames(self, camera_idx, source, request: Request,role:bool):
        """Generate frames from a specific camera feed"""
        if not self.is_connection_alive(source):
            logging.warning(f"[Camera {camera_idx}] Connection not available")
            return

        check_interval = 60  # seconds
        last_check = 0
        counter = 0

        def open_capture(source):
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                # Set buffer size to reduce latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                return cap
            return None

        # Retry logic for opening capture
        retries = 0
        cap = None

        while cap is None and retries < self.RETRY_LIMIT:
            cap = open_capture(source)
            if cap is None:
                logging.error(
                    f"[Camera {camera_idx}] Failed to open source. "
                    f"Retrying ({retries + 1}/{self.RETRY_LIMIT})..."
                )
                await asyncio.sleep(self.RETRY_DELAY)
                retries += 1

        if cap is None:
            logging.error(
                f"[Camera {camera_idx}] Could not open source after {self.RETRY_LIMIT} retries."
            )
            return

        fresh = FreshestFrame(cap)

        try:
            while fresh.is_alive() and not self._shutdown_event.is_set():
                now = time.time()

                # Periodic connection check
                if now - last_check >= check_interval:
                    if not self.is_connection_alive(source):
                        logging.warning(
                            f"[Camera {camera_idx}] Connection lost")
                        break
                    last_check = now

                # Check if client disconnected
                if await request.is_disconnected():
                    logging.info("Client disconnected, releasing camera.")
                    break

                success, frame = fresh.read()
                counter += 1

                if not success or frame is None:
                    # Generate blank frame if we can't read from camera
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(
                        frame, "No signal", (220, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
                    )
                else:
                    # Store original dimensions
                    original_height, original_width = frame.shape[:2]

                    # Process frame
                    if role == True:
                        frame=frame
                    else:

                        frame = await self.process_frame(frame, f'/rt{camera_idx}', counter)

                    # Resize back to original dimensions
                    frame = cv2.resize(
                        frame, (original_width, original_height))

                # Encode and yield the frame
                try:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                except Exception as e:
                    logging.error(f"Error encoding frame: {e}")
                    continue

        except Exception as e:
            logging.error(f"Error in generate_frames: {e}")
        finally:
            self.release_resources(fresh, cap,role)


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


def image_crop(filepath, isSearch):
    """Crop face from image with padding"""
    if isSearch:
        frame = cv2.imread(filepath)
        _, img_encoded = cv2.imencode(".jpg", frame)
        return img_encoded
    try:
        face_handler = FaceAnalysis(
            'antelopev2',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            root='.'
        )
        face_handler.prepare(ctx_id=0)

        frame = cv2.imread(filepath)
        if frame is None:
            raise ValueError(f"Could not load image: {filepath}")

        faces = face_handler.get(frame)
        if not faces:
            raise ValueError("No faces detected in image")

        facebox = faces[0].bbox
        x1, y1, x2, y2 = map(int, facebox)

        height_f, width_f = frame.shape[:2]
        padding = 40
        x1 = max(x1 - padding, 0)
        y1 = max(y1 - padding, 0)
        x2 = min(x2 + padding, width_f)
        y2 = min(y2 + padding, height_f)

        cropped_frame = frame[y1:y2, x1:x2]
        _, img_encoded = cv2.imencode(".jpg", cropped_frame)
        return img_encoded

    except Exception as e:
        logging.error(f"Error in image_crop: {e}")
        return None


if __name__ == "__main__":
    a = CCtvMonitor()
    embeddings, filenames = a.load_embeddings()

    query_path = 'outputs/screenshot/s.unknown_29.jpg'
    query_embedding = a.get_embedding(query_path)
    results = a.find_similar_images(
        query_embedding, embeddings, filenames, top_k=10)
    print("\nTop similar images:")
    for fname, score in results:
        print(f"{fname}: {score:.4f}")
