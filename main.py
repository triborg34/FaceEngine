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
from camera import FreshestFrame
from savatoDb import load_embeddings_from_db
# --- Basic Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CCTV-Server")
logging.getLogger('torch').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# --- Face Analysis Setup ---
face_handler = FaceAnalysis('buffalo_l', providers=[
                            'CUDAExecutionProvider', 'CPUExecutionProvider'])
face_handler.prepare(ctx_id=0)


RTSP_URL = "rtsp://admin:123456@192.168.1.245:554/stream"

# Environment variables for configuration
WEBSOCKET_HOST = os.getenv("WEBSOCKET_HOST", "127.0.0.1")  # WebSocket host
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 5000))    # WebSocket port
MODEL_PATH = os.getenv("MODEL_PATH", "models/yolov8n.pt")  # Path to YOLO model

# Load known face embeddings
known_names = {
}

try:
    known_names = load_embeddings_from_db()
except Exception as e:
    print(e)


# Recognition Thread Setup
recognition_queue = queue.Queue()
face_info = {}  # {face_id: {'name': str, 'bbox': (x1, y1, x2, y2)}}
face_info_lock = threading.Lock()


#
model = YOLO(MODEL_PATH, verbose=True)
cap = cv2.VideoCapture(RTSP_URL)
freshest = FreshestFrame(cap)
assert cap.isOpened()


# --- Functions ---


def update_face_info(face_id, name, bbox=None):
    with face_info_lock:
        face_info[face_id] = {'name': name, 'bbox': bbox}


def recognize_face(embedding):
    for name, embeds in known_names.items():
        for known_emb in embeds:
            sim = cosine_similarity([embedding], [known_emb])[0][0]
            if sim > 0.7:
                return name, sim
    return 'unknown', 0.0


def recognition_worker():
    logger.info("Recognition thread started.")
    while True:
        item = recognition_queue.get()
        if item is None:
            break
        face_id, face_img = item
        faces = face_handler.get(face_img)

        if faces:
            face = faces[0]
            name, sim = recognize_face(face.embedding)
            x1, y1, x2, y2 = map(int, face.bbox)
            update_face_info(face_id, f"{name} ({sim:.2f})", (x1, y1, x2, y2))
        else:
            update_face_info(face_id, "Unknown", None)

# --- Main Code ---


threading.Thread(target=recognition_worker, daemon=True).start()


async def main(websocket):
    try:
        counter = 0
        start_time = time.time()

        while True:
            try:
                ret, frame = freshest.read()
                if not ret or frame.size == 0:
                    continue

                frame = cv2.resize(frame, (640, 640))
                counter += 1

                results = model.predict(frame, classes=[0])  # Class 0 = person

                if results and len(results[0].boxes) > 0:
                    for i, box in enumerate(results[0].boxes):
                        x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                        human_crop = frame[y1:y2, x1:x2]

                        # Draw human bounding box
                        cv2.rectangle(frame, (x1, y1),
                                      (x2, y2), (0, 255, 0), 2)

                        # Push to recognition queue every 25 frames
                        if counter % 25 == 0:
                            try:
                                recognition_queue.put((i, human_crop))
                            except Exception as e:
                                logger.error(f"Queue error: {e}")

                        # Get recognition info
                        info = face_info.get(
                            i, {'name': "Unknown", 'bbox': None})
                        label = info['name']
                        face_bbox = info['bbox']

                        # Draw face bbox if exists
                        if face_bbox:
                            fx1, fy1, fx2, fy2 = face_bbox
                            cv2.rectangle(frame, (x1 + fx1, y1 + fy1),
                                          (x1 + fx2, y1 + fy2), (0, 0, 255), 2)
                            cv2.putText(frame, label, (x1 + fx1, y1 + fy1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        else:
                            cv2.putText(frame, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                # FPS calculation
                fps = 1.0 / (time.time() - start_time)
                start_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('frame', frame)
                _, encoded = cv2.imencode(
                    '.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                data = base64.b64encode(encoded).decode('utf-8')
                await websocket.send(data)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            except queue.Empty as e:
                logger.warning("Buffer is empty. Retrying...")
                continue

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
    """
    Handle WebSocket connections and start the main loop.
    """
    try:
        await main(websocket)
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"WebSocket connection closed: {e}")
    finally:
        logger.info("WebSocket handler stopped.")

# ---- WebSocket Server ----


async def websocket_server():
    """
    Start the WebSocket server.
    """
    logger.info(
        f"Starting WebSocket server at ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    server = await websockets.serve(
        ws_handler,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT,
    )
    await asyncio.Future()  # Run forever


if __name__ == "__main__":
    logger.info(
        f"Starting WebSocket server at ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    asyncio.run(websocket_server())
