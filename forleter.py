
# Import necessary libraries
from asyncio.log import logger
import base64
import os
import time
import cv2
import numpy as np
import queue
import requests
from ultralytics import YOLO
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity
import asyncio
import websockets
from threading import Lock

from camera import FreshestFrame


RTSP_URL="rtsp://admin:123456@192.168.1.245:554/stream"

# Environment variables for configuration
WEBSOCKET_HOST = os.getenv("WEBSOCKET_HOST", "127.0.0.1")  # WebSocket host
WEBSOCKET_PORT = int(os.getenv("WEBSOCKET_PORT", 5000))    # WebSocket port
MODEL_PATH = os.getenv("MODEL_PATH", "./yolov8n-face.pt")  # Path to YOLO model

# ---- Load YOLOv8-Face ----
# Initialize YOLO model for face detection
yolo_face = YOLO('yolov8n-face.pt')

# Specify the path to the downloaded models
model_dir = "./"  # Replace with the actual path

# Initialize FaceAnalysis for face embedding extraction
face_embedder = FaceAnalysis(name='buffalo_l', root=model_dir, providers=['CPUExecutionProvider'])
face_embedder.prepare(ctx_id=0)

# ---- Load known faces ----
# Dictionary to store known faces and their embeddings
known_faces = {}  # name -> list of embeddings

# Function to reshape embeddings into the required dimension
def safe_reshape(embedding, dim=512):
    """
    Reshape a flat embedding list into a nested list of vectors with the specified dimension.
    """
    if isinstance(embedding[0], list) and len(embedding[0]) == dim:
        return embedding
    
    if len(embedding) % dim != 0:
        raise ValueError(f"Inconsistent embedding length: {len(embedding)} not divisible by {dim}")
    
    return [embedding[i:i+dim] for i in range(0, len(embedding), dim)]

# Function to load embeddings from a database
def load_embeddings_from_db():
    """
    Load known face embeddings from a database and store them in the `known_faces` dictionary.
    """
    url = "http://127.0.0.1:8090/api/collections/known_face/records?perPage=1000"

    try:
        res = requests.get(url)
        res.raise_for_status()
        records = res.json()["items"]

        for item in records:
            name = item["name"]
            embedding = item.get("embdanings")
            if embedding:
                embedding = embedding[:len(embedding) - (len(embedding) % 512)]
                try:
                    reshaped = safe_reshape(embedding)
                    for emb in reshaped:
                        emb_array = np.array(emb, dtype=np.float32)
                        known_faces.setdefault(name, []).append(emb_array)
                except Exception as reshape_error:
                    print(f"⚠️ Error reshaping embedding for {name}: {reshape_error}")
        
        print(f"✅ Loaded {sum(len(v) for v in known_faces.values())} embeddings from {len(known_faces)} persons")

    except Exception as e:
        print(f"❌ Failed to load embeddings: {e}")

# Load known faces at startup
try:
    load_embeddings_from_db()
    print("Known faces loaded.")
except Exception as e:
    print("Error loading faces:", e)

# ---- RTSP or webcam ----
# Initialize video capture (default is webcam)
cap = cv2.VideoCapture(RTSP_URL)  # Change to RTSP link if needed
fresh=FreshestFrame(cap)
cap.set(cv2.CAP_PROP_FPS, 30)

# ---- Recognition ----
# Queue to store face crops for recognition
recognition_queue = queue.Queue()
# Dictionary to store recognized face names
face_names = {}
# Lock for thread-safe access to `face_names`
face_names_lock = Lock()

# Function to update recognized face names
def update_face_names(face_id, name):
    """
    Update the `face_names` dictionary in a thread-safe manner.
    """
    with face_names_lock:
        face_names[face_id] = name

# Removed threading functionality for recognition
# Updated recognize function to handle recognition directly without threading

def recognize(face_crop):
    """
    Process a face crop and recognize the face.
    """
    faces = face_embedder.get(face_crop)
    print(faces)
    if faces:
        embedding = faces[0].embedding
        name, sim = "Unknown", 0
        for known_name, embeddings in known_faces.items():
            for known_emb in embeddings:
                sim = cosine_similarity([embedding], [known_emb])[0][0]
                if sim > 0.6:  # Threshold
                    name = known_name
                    break
            if name != "Unknown":
                break
        return f"{name} ({sim:.2f})"
    return "Unknown"

# ---- Main loop ----
async def mainLoop(websocket):
    """
    Main loop to process frames, detect faces, and send results via WebSocket.
    """
    try:
        frame_id = 0
        start_time = time.time()
        while True:
            try:
                ret, frame = fresh.read()
                if not ret or frame is None or frame.size == 0:
                    logger.warning("Lost connection or empty frame from RTSP. Reconnecting...")
                    break

                frame_id += 1
                logger.info(f"Processing frame {frame_id}")
                results = yolo_face(frame)[0]

                if results.boxes is not None and len(results.boxes) > 0:
                    boxes = results.boxes.data.cpu().numpy()

                    for i, box in enumerate(boxes):
                        x1, y1, x2, y2, conf = box[:5]
                        if conf < 0.5:
                            continue

                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

                        # Padding around face
                        padding = 100
                        h, w, _ = frame.shape
                        x1 = max(0, x1 - padding)
                        y1 = max(0, y1 - padding)
                        x2 = min(w, x2 + padding)
                        y2 = min(h, y2 + padding)

                        face_crop = frame[y1:y2, x1:x2]
                        

                        # Only recognize every 25 frames
                        if frame_id % 25 == 0:
                            label = recognize(face_crop)
                            update_face_names(i, label)

                        label = face_names.get(i, "Unknown")

                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                fps = 1.0 / (time.time() - start_time)
                start_time = time.time()
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("YOLO-Face Recognition", frame)
                _, encoded = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                data = base64.b64encode(encoded).decode('utf-8')
                await websocket.send(data)
                try:
                    pass
                    # del frame, face_crop, results, boxes
                except Exception as e:
                    print(e)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            except queue.Empty:
                logger.warning("Buffer is empty. Retrying...")
                continue

    except websockets.exceptions.ConnectionClosed as e:
        print(f"WebSocket connection closed: {e}")
    except Exception as e:
        logger.error(f"Error in frame processing: {e}")
    finally:
        recognition_queue.put(None)
        if cap.isOpened():
            cap.release()
            
        cv2.destroyAllWindows()
        fresh.release()
        
        logger.info("Resources released and application closed.")

# ---- WebSocket Handler ----
async def ws_handler(websocket):
    """
    Handle WebSocket connections and start the main loop.
    """
    try:
        await mainLoop(websocket)
    except websockets.exceptions.ConnectionClosed as e:
        logger.info(f"WebSocket connection closed: {e}")
    finally:
        logger.info("WebSocket handler stopped.")

# ---- WebSocket Server ----
async def websocket_server():
    """
    Start the WebSocket server.
    """
    logger.info(f"Starting WebSocket server at ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")
    server = await websockets.serve(
        ws_handler,
        WEBSOCKET_HOST,
        WEBSOCKET_PORT,
    )
    await asyncio.Future()  # Run forever

# ---- Main Entry Point ----
if __name__ == "__main__":
    asyncio.run(websocket_server())
    logger.info(f"Starting WebSocket server at ws://{WEBSOCKET_HOST}:{WEBSOCKET_PORT}")