import asyncio
import base64
import logging
import os
import queue
import time
import warnings
import threading
import cv2
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import websockets
import torch
from concurrent.futures import ThreadPoolExecutor
from camera import FreshestFrame
from savatoDb import load_embeddings_from_db

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CCTV-Server")
logging.getLogger('torch').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# frps = 5 if device == 'cuda' else 25

face_handler = FaceAnalysis('buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
face_handler.prepare(ctx_id=0)

RTSP_URL = "rtsp://admin:123456@192.168.1.245:554/stream"
WEBSOCKET_HOST = os.getenv("WEBSOCKET_HOST", "127.0.0.1")
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 5000))
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")


track_last_seen = {}
RECOGNITION_REFRESH_INTERVAL = 1  # seconds
# Load known embeddings
try:
    known_names = load_embeddings_from_db()
except Exception as e:
    print(e)
    known_names = {}

recognition_queue = queue.Queue()
face_info = {}    # {track_id: {'name': str, 'bbox': (x1,y1,x2,y2), 'last_update': time.time()}}
face_info_lock = threading.Lock()
embedding_cache = {}  # Optional cache {track_id: embedding}

model = YOLO(MODEL_PATH, verbose=False)
cap = cv2.VideoCapture(RTSP_URL)
freshest = FreshestFrame(cap)
assert cap.isOpened()

executor = ThreadPoolExecutor(max_workers=4)  # Recognition threads


# --- Functions ---

def update_face_info(track_id, name, bbox=None):
    with face_info_lock:
        face_info[track_id] = {'name': name, 'bbox': bbox, 'last_update': time.time()}


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
        print(best_score)
        return best_match, best_score
    else:
        return "unknown", best_score


def recognition_worker():
    logger.info("Recognition thread started.")
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
            update_face_info(track_id, name, (x1, y1, x2, y2))
            embedding_cache[track_id] = face.embedding
        else:
            update_face_info(track_id, "Unknown", None)


# --- Threads ---

threading.Thread(target=recognition_worker, daemon=True).start()


# --- Main Loop ---

async def main(websocket):
    try:
        counter = 0
        start_time = time.time()

        while True:
            ret, frame = freshest.read()
            if not ret or frame.size == 0:
                continue

            frame = cv2.resize(frame, (640, 640))
            counter += 1

            results = model.track(frame, classes=[0], tracker="bytetrack.yaml", persist=True, device=device)

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
                    current_time = time.time()
                    last_seen = track_last_seen.get(track_id, 0)
                    if (current_time - last_seen > RECOGNITION_REFRESH_INTERVAL) or (track_id not in track_last_seen):
                        try:
                            recognition_queue.put((track_id, human_crop))
                            track_last_seen[track_id] = current_time
                        except Exception as e:
                            logger.error(f"Queue error: {e}")
                            continue
                        

                    info = face_info.get(track_id, {'name': "Unknown", 'bbox': None})
                    label = f"{info['name']} ID:{track_id}"
                    face_bbox = info['bbox']

                    if face_bbox:
                        fx1, fy1, fx2, fy2 = face_bbox
                        cv2.rectangle(frame, (x1 + fx1, y1 + fy1), (x1 + fx2, y1 + fy2), (0, 0, 255), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # FPS calc
            fps = 1.0 / (time.time() - start_time)
            start_time = time.time()
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # WebSocket sending
            _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            data = base64.b64encode(encoded).decode('utf-8')
            await websocket.send(data)

    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"WebSocket connection closed: {e}")
    except Exception as e:
        logger.error(f"Error in main processing: {e}")
    finally:
        recognition_queue.put(None)
        freshest.release()
        cap.release()
        cv2.destroyAllWindows()


async def ws_handler(websocket):
    try:
        await main(websocket)
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"WebSocket connection closed: {e}")
    finally:
        logger.info("WebSocket handler stopped.")


async def websocket_server():
    logger.info(f"Starting WebSocket server at ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    server = await websockets.serve(ws_handler, WEBSOCKET_HOST, WEBSOCKET_PORT)
    await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(websocket_server())
