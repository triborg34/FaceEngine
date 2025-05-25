import asyncio
import gc
import logging
import multiprocessing
import os
import queue
import time
import threading
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

# --- Basic Setup ---
logging.getLogger('torch').setLevel(logging.ERROR)
# warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
logging.basicConfig(
    level=logging.DEBUG,  # Capture everything from DEBUG and above

    format='[%(asctime)s] [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("log.txt", mode='a',
                            encoding='utf-8'),  # Append mode
        logging.StreamHandler()  # Optional: also show logs in console
    ]
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
frps = 5 if device == 'cuda' else 25

face_handler = FaceAnalysis('buffalo_l', providers=[
                            'CUDAExecutionProvider', 'CPUExecutionProvider'])
face_handler.prepare(ctx_id=0)
cv2.setNumThreads(multiprocessing.cpu_count())
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")
# Frame rate limiter (FPS)
TARGET_FPS = 30  # Adjust based on your needs
FRAME_DELAY = 1.0 / TARGET_FPS
# Constants for health monitoring
RETRY_LIMIT = 5
RETRY_DELAY = 3  # seconds
model = None

# Load known embeddings
try:
    known_names = load_embeddings_from_db()
except Exception as e:
    logging.error(e)
    known_names = {}

lock = threading.Lock()
recognition_queue = queue.Queue()
# {track_id: {'name': str, 'bbox': (x1,y1,x2,y2), 'last_update': time.time()}}
face_info = {}
face_info_lock = threading.Lock()
embedding_cache = {}  # Optional cache {track_id: embedding}


executor = ThreadPoolExecutor(max_workers=4)  # Recognition threads


# --- Functions ---

def realseFreshest(fresh: FreshestFrame, cap: cv2.VideoCapture):
    try:
        fresh.release()
        cap.release()
        recognition_queue.put(None)

    except Exception as e:
        logging.error(f"Error to Realse Cameras : {e}")


def loadModel():
    global model
    if model == None:
        model = YOLO(MODEL_PATH, verbose=False)
        model.eval()
        logging.info('Model Load.')
    with lock:
        return model


def graceful_shutdown():

    # Clean up resources
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Stop observer if running

    logging.info("Cleanup complete. Shutting down.")

    os._exit(0)  # Use os._exit instead of sys.exit for more forceful termination


def update_face_info(track_id, name, score, bbox=None):
    with face_info_lock:
        face_info[track_id] = {'name': name, 'bbox': bbox,
                               'last_update': time.time(), 'score': score}


def recognize_face(embedding):
    best_match = 'unknown'
    best_score = 0.0
    for name, embeds in known_names.items():
        for known_emb in embeds:
            sim = cosine_similarity([embedding], [known_emb])[0][0]
            if sim > best_score:
                best_score = sim
                best_match = name
    if best_score >= 0.6:

        return best_match, best_score
    else:
        return "unknown", best_score


def recognition_worker():
    logging.info("Recognition thread started.")
    while True:
        item = recognition_queue.get()
        if item is None:
            break
        track_id, face_img = item

        # Optional caching: skip if recently updated
        if track_id in face_info and time.time() - face_info[track_id]['last_update'] < 2:
            continue

        faces = face_handler.get(face_img)
        if faces:
            face = faces[0]
            name, sim = recognize_face(face.embedding)
            x1, y1, x2, y2 = map(int, face.bbox)
            update_face_info(track_id, name, sim, (x1, y1, x2, y2))
            embedding_cache[track_id] = face.embedding
        else:
            update_face_info(track_id, "Unknown", None)


# --- Threads ---

threading.Thread(target=recognition_worker, daemon=True).start()


# --- Main Loop ---

async def process_frame(frame, path,counter):
    
    try:
        
        
        start_time = time.time()

        while True:
            if frame.size == 0:
                continue

            frame = cv2.resize(frame, (640, 640))
            

            results = model.track(
                frame, classes=[0], tracker="bytetrack.yaml", persist=True, device=device)

            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0][:4].cpu().tolist())
                    try:
                        track_id = int(box.id[0].cpu().tolist())
                    except Exception:
                        continue

                    human_crop = frame[y1:y2, x1:x2]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Only recognize every frps frames
                    if counter % frps == 0:
                        recognition_queue.put((track_id, human_crop))

                    info = face_info.get(
                        track_id, {'name': "Unknown", "score": 0, 'bbox': None})
                    label = f"{info['name']} ID:{track_id}"
                    face_bbox = info['bbox']
                    try:
                        score = int(info['score']*100)
                    except TypeError:
                        score = 0
                    name = info['name']

                    if face_bbox:
                        fx1, fy1, fx2, fy2 = face_bbox
                        cv2.rectangle(frame, (x1 + fx1, y1 + fy1),
                                      (x1 + fx2, y1 + fy2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        try:
                            await insertToDb(name, frame, human_crop,
                                       score, track_id)
                        except Exception as e:
                            logging.error(f"Error Insert to DB {e}")
                            continue

                    else:
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # FPS calc
            fps = 1.0 / (time.time() - start_time)
            start_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            return frame
    except Exception as e:
        return frame


async def generate_frames(camera_idx, source, request: Request):
    """Generate frames from a specific camera feed"""
    loadModel()

    def open_capture(source):
        cap = cv2.VideoCapture(source)
        return cap if cap.isOpened() else None

    retries = 0
    cap = open_capture(source)

    while cap is None and retries < RETRY_LIMIT:
        logging.error(
            f"[Camera {camera_idx}] Failed to open source. Retrying ({retries + 1}/{RETRY_LIMIT})...")
        await asyncio.sleep(RETRY_DELAY)
        retries += 1
        cap = open_capture(source)

    if cap is None:
        logging.error(
            f"[Camera {camera_idx}] Could not open source after {RETRY_LIMIT} retries.")
        return
    fresh = FreshestFrame(cap)

    try:
        while fresh.is_alive():

            if await request.is_disconnected():
                logging.info("Client disconnected, releasing camera.")
                break
            success, frame = fresh.read()
            height,width=frame.shape[0],frame.shape[1]
            if not success:

                # Generate blank frame if we can't read from camera
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No signal", (220, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:

                frame =await process_frame(frame, f'/rt{camera_idx}',success)
            
            frame=cv2.resize(frame,(width,height))

            # Encode and yield the frame
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    finally:
        realseFreshest(fresh, cap)
