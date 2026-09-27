
import asyncio
import json
import logging
import os

import torch
os.environ['ULTRALYTICS_SKIP_REQUIREMENTS_CHECKS'] = '1'
import cv2
import shutil
import socket
import threading
import time
import base64
from contextlib import asynccontextmanager
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import FastAPI, File, Query, Request, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests
import uvicorn
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
# Import your improved CCtvMonitor class
from newengine import (
    CCtvMonitor, image_crop, CameraManager, sendRegularFrames, takeFrame,
    _get_crop_face_handler, _crop_face_lock, FACE_CROP_PADDING,
)
from onvifmaneger import get_rtsp_url
import urllib.request

from newsavatoDb import (
    reciveFromUi, reciveFromUi_multi, init_db_session,
    shutdown_relay_executor, get_db_worker,
    add_embedding_to_person, remove_person_embedding,
    delete_person_from_db, get_person_faces,
    extract_face_embedding, validate_face_embedding,
    build_embedding_meta, face_blur_score, face_yaw, get_min_face_px,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

camera_registry = {}
camera_registry_lock = threading.Lock()
cctv_monitor = None
class RtspFields(BaseModel):
    ip: str
    port: str
    username: str
    password: str


class KnownPersonFields(BaseModel):
    name: str
    gender: str
    imagePath: str
    age: str
    role: str
    socialnumber: str
    userwhom: str = ""
    description: str = ""


class MultiImagePersonFields(BaseModel):
    name: str
    gender: str
    imagePaths: list[str]
    age: str
    role: str
    socialnumber: str
    userwhom: str = ""
    description: str = ""


class AddFaceReferenceFields(BaseModel):
    name: str
    imagePath: str


class RemoveFaceReferenceFields(BaseModel):
    name: str
    embeddingIndex: int


class RelayConfig(BaseModel):
    ip: str
    port: int
    username: str
    password: str


class TakePicture(BaseModel):
    rtspUrl:str
    camName:str

class RegisterFaceFields(BaseModel):
    """Register a specific detected face to a person."""
    filePath: str
    faceIndex: int
    name: str
    gender: str
    age: str
    role: str
    socialnumber: str

# Global CCTV monitor instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global cctv_monitor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cctv_monitor = CCtvMonitor(device=device)

    # Startup
    logging.info("Starting CCTV Monitor application...")
    try:

        # Warm up the PocketBase HTTP connection pool
        init_db_session()

        # Initialise the dedicated DbWorker so all DB inserts are async
        get_db_worker()

        logging.info("CCTV Monitor initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize CCTV Monitor: {e}")
        raise

    yield

    # Shutdown
    logging.info("Shutting down CCTV Monitor application...")
    if cctv_monitor:
        await cctv_monitor.graceful_shutdown()
    shutdown_relay_executor()
    logging.info("Application shutdown complete")

# Create FastAPI app with lifespan manager
app = FastAPI(lifespan=lifespan)

# CORS configuration
origins = ["*"]  # Change this to specific domains in production

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Origin", "X-Requested-With", "Content-Type", "Accept"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    db_stats = get_db_worker().stats if cctv_monitor else {}
    return {
        "status": "healthy",
        "cctv_monitor_active": cctv_monitor is not None,
        "timestamp": time.time(),
        "multiprocessing": multiprocessing.cpu_count(),
        "db_worker": db_stats,
    }


@app.get("/{camera_id}")
async def video_feed(
    camera_id: str,
    request: Request,
    source: str = Query(...),
    role: bool = Query(False)
):
    
    if role:
        return StreamingResponse(
        sendRegularFrames(source,request),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )
    if source == "0":
        source = int(source)
    camera_idx = int(camera_id[2:])
    # get or create camera
    with camera_registry_lock:
        if source not in camera_registry:
            camera_registry[source] = CameraManager(source, cctv_monitor,camera_idx)

        cam = camera_registry[source]

    cam.add_client()

    async def watch_disconnect():
        while True:
            if await request.is_disconnected():
                cam.remove_client()
                # Drop the manager from the registry once it has no clients
                # so stopped cameras don't accumulate
                if not cam.has_clients():
                    with camera_registry_lock:
                        if camera_registry.get(source) is cam:
                            del camera_registry[source]
                break
            await asyncio.sleep(0.2)

    asyncio.create_task(watch_disconnect())

    return StreamingResponse(
        cam.sendFrames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )




def _probe_port(ip, port=80, timeout=0.3):
    """Return ip if the port is open, else None"""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return ip if sock.connect_ex((ip, port)) == 0 else None
    except Exception as e:
        logging.debug(f"Error scanning {ip}: {e}")
        return None


def _discover_worker(result_q: Queue, timeout: float = 0.3):
    """Run the parallel port scan in a worker thread, pushing found IPs."""
    ip_base = "192.168.1"
    ips = [f"{ip_base}.{i}" for i in range(1, 255)]
    with ThreadPoolExecutor(max_workers=64) as pool:
        futures = {pool.submit(_probe_port, ip, 80, timeout): ip for ip in ips}
        for fut in as_completed(futures):
            ip = fut.result()
            if ip:
                result_q.put(ip)


async def discover_onvif_stream():
    """Discover ONVIF cameras on the network (parallel scan, off event loop)"""
    loop = asyncio.get_running_loop()
    result_q: Queue = Queue()
    worker = threading.Thread(
        target=_discover_worker, args=(result_q,), daemon=True)
    worker.start()

    while worker.is_alive() or not result_q.empty():
        try:
            ip = await loop.run_in_executor(None, result_q.get, True, 0.2)
        except Empty:
            continue
        yield f"data: {json.dumps({'ip': ip, 'port': 80, 'status': 'found'})}\n\n"


@app.get("/onvif/get-stream")
async def get_camera_stream():
    """Stream ONVIF camera discovery results"""
    return StreamingResponse(
        discover_onvif_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@app.post('/onvif/get-rtsp')
def get_camera_rtsp(request: RtspFields):
    """Get RTSP URL from ONVIF camera"""
    try:
        logging.info(
            f"Getting RTSP URL for camera at {request.ip}:{request.port}")

        port = int(request.port)
        rtsp_url = get_rtsp_url(
            request.ip, port, request.username, request.password)

        if not rtsp_url:
            raise HTTPException(
                status_code=404, detail="Could not retrieve RTSP URL")

        return {'rtsp': rtsp_url}

    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid port number")
    except Exception as e:
        logging.error(f"Error getting RTSP URL: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to get RTSP URL: {str(e)}")




@app.post('/takePicture')
def takePicture(request:TakePicture):
    rtspUrl = request.rtspUrl
    camName = request.camName
    nowSec = int(time.time())
    filename = f"{camName}_{nowSec}.jpg"
    UPLOAD_DIR_VIDEO = "uploads"
    os.makedirs(UPLOAD_DIR_VIDEO, exist_ok=True)
    file_location = os.path.join(UPLOAD_DIR_VIDEO, filename)
    try:
        img_encoded = takeFrame(rtspUrl, file_location)
        if img_encoded is None:
            raise HTTPException(
                status_code=400, detail="No face detected in image")
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        return {
            "success": True,
            "file_location": file_location,
            "filename": filename,
            "image_data": img_base64,
            "media_type": "image/jpeg"
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error taking picture: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to capture frame: {str(e)}")


@app.post("/upload")
def upload_file(isSearch: bool, file: UploadFile = File(...)):
    """Upload and process image file"""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}"
        )

    UPLOAD_DIR_VIDEO = "uploads"
    os.makedirs(UPLOAD_DIR_VIDEO, exist_ok=True)

    # Generate unique filename to avoid conflicts
    timestamp = int(time.time())
    filename = f"{timestamp}_{file.filename}"
    file_location = os.path.join(UPLOAD_DIR_VIDEO, filename)

    try:
        # Save uploaded file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logging.info(f"File uploaded: {file_location}")

        # Process image to crop face

        img_encoded = image_crop(file_location, isSearch)

        if img_encoded is None:
            # Clean up file if processing failed
            os.remove(file_location)
            raise HTTPException(
                status_code=400, detail="No face detected in image")

        # Convert image to base64
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')

        return {
            "success": True,
            "file_location": file_location,
            "filename": filename,
            "image_data": img_base64,
            "media_type": "image/jpeg"
        }

    except HTTPException:
        # Clean up file if processing failed
        if os.path.exists(file_location):
            os.remove(file_location)
        raise
    except Exception as e:
        # Clean up file if processing failed
        if os.path.exists(file_location):
            os.remove(file_location)
        logging.error(f"Error processing uploaded file: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


def _detect_faces_two_stage(frame):
    """Detect faces using YOLO person detection → InsightFace face detection.

    Runs YOLOv8 to find person bounding boxes, crops each person region,
    then runs InsightFace on each crop to find faces.  Returns a list of
    face dicts with bbox coordinates in the original full-image space.
    """
    detected_faces = []
    min_face_px = get_min_face_px(cctv_monitor)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # cv2.imshow('frame',frame)
    # cv2.waitKey(0)

    # Stage 1: YOLO person detection
    if cctv_monitor and cctv_monitor.model is not None:
        with cctv_monitor.model_lock:
            yolo_results = cctv_monitor.model.predict(frame, classes=[0], device=device)
        person_boxes = []
        for r in yolo_results:
            if r.boxes is not None:
                for box in r.boxes:
               
                    x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                    person_boxes.append((x1, y1, x2, y2))
        # Fallback: if YOLO found no persons, try the full image
        if not person_boxes:
            h, w = frame.shape[:2]
            person_boxes = [(0, 0, w, h)]
    else:
        # Fallback: no YOLO available, treat entire image as one region
        h, w = frame.shape[:2]
        person_boxes = [(0, 0, w, h)]

    # Stage 2: InsightFace on each person crop
    face_handler = _get_crop_face_handler()
    global_idx = 0

    for (px1, py1, px2, py2) in person_boxes:
        # Clamp to image bounds
        px1 = max(px1, 0)
        py1 = max(py1, 0)
        px2 = min(px2, frame.shape[1])
        py2 = min(py2, frame.shape[0])
        if px2 <= px1 or py2 <= py1:
            continue

        person_crop = frame[py1:py2, px1:px2]
       
        with _crop_face_lock:
            faces = face_handler.get(person_crop)

        if not faces:
            continue

        for face in faces:
            fx1, fy1, fx2, fy2 = map(int, face.bbox)
            face_w = fx2 - fx1
            face_h = fy2 - fy1


            if min(face_w, face_h) < min_face_px:
                continue

            # Convert face bbox from person-crop coords to full-image coords
            abs_fx1 = px1 + fx1
            abs_fy1 = py1 + fy1
            abs_fx2 = px1 + fx2
            abs_fy2 = py1 + fy2

            # Crop face with padding for the preview thumbnail
            pad = 40
            cx1 = max(abs_fx1 - pad, 0)
            cy1 = max(abs_fy1 - pad, 0)
            cx2 = min(abs_fx2 + pad, frame.shape[1])
            cy2 = min(abs_fy2 + pad, frame.shape[0])
            cropped = frame[cy1:cy2, cx1:cx2]
          
            _, crop_encoded = cv2.imencode(".jpg", cropped)
            crop_b64 = base64.b64encode(crop_encoded.tobytes()).decode('utf-8')

            detected_faces.append({
                "index": global_idx,
                "bbox": [abs_fx1, abs_fy1, abs_fx2, abs_fy2],
                "det_score": round(float(face.det_score), 4),
                "gender": 'female' if face.gender == 0 else 'male',
                "age": int(face.age),
                "crop_image": crop_b64,
            })
            global_idx += 1

    return detected_faces


@app.post("/detect-faces")
def detect_faces(file: UploadFile = File(...)):
    """Detect ALL faces in an uploaded image and return their details.

    Unlike /upload (which only returns the first face crop), this endpoint
    returns every face found: bounding boxes, cropped face images, detection
    scores, and image dimensions.  Used by the Flutter face-selection UI.

    Two-stage pipeline: YOLO detects person regions, then InsightFace
    detects faces within each person crop.  Bounding boxes are returned
    in full-image coordinates.
    """
    logging.error(file.filename)
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    logging.error("HELLLLLLLOOOOOOOOOOOOOOOOOOOO")

    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(allowed_extensions)}")

    UPLOAD_DIR_VIDEO = "uploads"
    os.makedirs(UPLOAD_DIR_VIDEO, exist_ok=True)

    timestamp = int(time.time())
    filename = f"{timestamp}_{file.filename}"
    file_location = os.path.join(UPLOAD_DIR_VIDEO, filename)

    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logging.info(f"detect-faces: processing {file_location}")

        frame = cv2.imread(file_location)

        if frame is None:
            # os.remove(file_location)
            raise HTTPException(status_code=400, detail="Could not read image")

        detected_faces = _detect_faces_two_stage(frame)

        if not detected_faces:
            # os.remove(file_location)
            logging.error(f"No face detected in image (faces must be at "
                       f"least {get_min_face_px(cctv_monitor)}px wide)")
            raise HTTPException(
              
                
                status_code=400,
                detail=f"No face detected in image (faces must be at "
                       f"least {get_min_face_px(cctv_monitor)}px wide)")

        height, width = frame.shape[:2]
        _, full_encoded = cv2.imencode(".jpg", frame)
        full_b64 = base64.b64encode(full_encoded.tobytes()).decode('utf-8')

        return {
            "success": True,
            "file_location": file_location,
            "filename": filename,
            "image_width": width,
            "image_height": height,
            "full_image": full_b64,
            "faces": detected_faces,
            "face_count": len(detected_faces),
        }

    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(file_location):
            os.remove(file_location)
        logging.error(f"Error in detect-faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/util/detect-faces-path")
def detect_faces_from_path(filePath: str):
    """Detect ALL faces in a server-side image file.

    Same as /detect-faces but accepts a file path instead of a multipart
    upload.  Used by the camera-capture flow where the full frame is
    already saved on disk.
    """
    filePath=r'{0}'.format(filePath)
    logging.info(f"{filePath=}")
    try:
        if not os.path.exists(filePath):
            raise HTTPException(status_code=400, detail="File not found")

        frame = cv2.imread(filePath)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read image")

        detected_faces = _detect_faces_two_stage(frame)

        if not detected_faces:
            raise HTTPException(
                status_code=400,
                detail=f"No face detected in image (faces must be at "
                       f"least {get_min_face_px(cctv_monitor)}px wide)")

        height, width = frame.shape[:2]
        _, full_encoded = cv2.imencode(".jpg", frame)
        full_b64 = base64.b64encode(full_encoded.tobytes()).decode('utf-8')

        return {
            "success": True,
            "file_location": filePath,
            "filename": os.path.basename(filePath),
            "image_width": width,
            "image_height": height,
            "full_image": full_b64,
            "faces": detected_faces,
            "face_count": len(detected_faces),
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in detect-faces-path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/register-face")
def register_selected_face(data: RegisterFaceFields):
    """Register a specific face (by index) from a previously detected set.

    The client first calls /detect-faces, the user picks a face, then
    this endpoint is called with the file path and face index.  The
    backend re-detects faces to validate the index and extracts the
    embedding for only the selected face.
    """
    try:
        if not cctv_monitor:
            raise HTTPException(
                status_code=503, detail="CCTV Monitor not initialized")

        if not data.name.strip():
            raise HTTPException(status_code=400, detail="Name is required")

        file_path = data.filePath
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=400, detail="Image file not found on server")

        frame = cv2.imread(file_path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not read image")

        # Re-run two-stage detection to get face embeddings (must match
        # the indices returned by /detect-faces).
        min_face_px = get_min_face_px(cctv_monitor)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Stage 1: YOLO person detection
        if cctv_monitor and cctv_monitor.model is not None:
            with cctv_monitor.model_lock:
                yolo_results = cctv_monitor.model(frame, classes=[0], device=device)
            person_boxes = []
            for r in yolo_results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0][:4])
                        person_boxes.append((x1, y1, x2, y2))
            # Fallback: if YOLO found no persons, try the full image
            if not person_boxes:
                h, w = frame.shape[:2]
                person_boxes = [(0, 0, w, h)]
        else:
            h, w = frame.shape[:2]
            person_boxes = [(0, 0, w, h)]

        # Stage 2: InsightFace on each person crop, collect valid faces
        face_handler = _get_crop_face_handler()
        valid_faces = []
        for (px1, py1, px2, py2) in person_boxes:
            px1, py1 = max(px1, 0), max(py1, 0)
            px2, py2 = min(px2, frame.shape[1]), min(py2, frame.shape[0])
            if px2 <= px1 or py2 <= py1:
                continue
            person_crop = frame[py1:py2, px1:px2]
            with _crop_face_lock:
                faces = face_handler.get(person_crop)
            if not faces:
                continue
            for face in faces:
                fx1, fy1, fx2, fy2 = map(int, face.bbox)
                if min(fx2 - fx1, fy2 - fy1) >= min_face_px:
                    # face.bbox is relative to the person crop; remember the
                    # crop origin so the absolute box can be rebuilt later.
                    valid_faces.append((face, px1, py1))

        if not valid_faces:
            raise HTTPException(
                status_code=400,
                detail="No face large enough detected in image")

        if data.faceIndex < 0 or data.faceIndex >= len(valid_faces):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid face index {data.faceIndex}. "
                       f"Valid range: 0-{len(valid_faces) - 1}")

        selected_face, crop_x, crop_y = valid_faces[data.faceIndex]
        embedding = selected_face.embedding

        # Quality record for this reference vector: stored next to
        # embdanings in embeddingMeta so a later audit can see how sharp /
        # frontal the photo that produced the vector actually was.
        ax1, ay1, ax2, ay2 = map(int, selected_face.bbox)
        fh, fw = frame.shape[:2]
        abs_x1, abs_y1 = crop_x + ax1, crop_y + ay1
        abs_x2, abs_y2 = crop_x + ax2, crop_y + ay2
        face_crop = frame[max(abs_y1, 0):min(abs_y2, fh),
                          max(abs_x1, 0):min(abs_x2, fw)]
        meta = build_embedding_meta(
            face_blur_score(face_crop),
            float(getattr(selected_face, "det_score", 0.0) or 0.0),
            face_yaw(selected_face))

        if not validate_face_embedding(embedding):
            raise HTTPException(
                status_code=400, detail="Invalid face embedding generated")

        # Fetch existing person data to preserve fields
        from newsavatoDb import find_person_record
        record = find_person_record(data.name)
        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"Person '{data.name}' not found. "
                       f"Create them first with /insertKToDp.")

        ok = add_embedding_to_person(
            data.name, embedding, file_path,
            record.get('age', ''), record.get('gender', ''),
            record.get('role', ''), record.get('socialnumber', ''),
            userwhom=record.get('userwhom', ''),
            description=record.get('description', ''),
            face_handler=_get_crop_face_handler(),
            face_lock=_crop_face_lock,
            model=cctv_monitor.model if cctv_monitor else None,
            model_lock=cctv_monitor.model_lock if cctv_monitor else None,
            meta=meta)

        if ok and cctv_monitor:
            cctv_monitor.refresh_person(data.name)

        return {
            "success": ok,
            "message": (
                f"Face added for '{data.name}'"
                if ok else "Duplicate face — not added"),
            "name": data.name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error registering selected face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/insertKToDp")
def insert_known_person(data: KnownPersonFields):
    """Insert known person data to database (single image).

    Runs synchronously in a background thread via FastAPI's threadpool so
    the event loop is never blocked.
    """
    try:
        if not cctv_monitor:
            raise HTTPException(
                status_code=503, detail="CCTV Monitor not initialized")

        # Validate required fields
        if not data.name.strip():
            raise HTTPException(status_code=400, detail="Name is required")

        if not data.imagePath.strip():
            raise HTTPException(
                status_code=400, detail="Image path is required")

        # Check if image path is URL or local path
        is_url = data.imagePath.startswith(('http://', 'https://'))

        logging.info(f"Inserting known person: {data.name} (URL: {is_url})")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Call database insertion function, reusing the already-loaded models
        result = reciveFromUi(
            data.name,
            data.imagePath,
            data.age,
            data.gender,
            data.role,
            data.socialnumber,
            is_url,
            device,
            cctv_monitor.face_handler,
            cctv_monitor.model,
            cctv_monitor.face_lock,
            cctv_monitor.model_lock,
            userwhom=data.userwhom,
            description=data.description,
            min_face_px=get_min_face_px(cctv_monitor),
        )

        # Refresh known names in CCTV monitor (incremental: only the
        # newly added person is reloaded, not the whole database)
        if cctv_monitor:
            cctv_monitor.refresh_person(data.name)
            logging.info("Known names refreshed in CCTV monitor,")

        return {
            "success": True,
            "message": "Person added successfully",
            "name": data.name
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error inserting known person: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/insertKToDpMulti")
def insert_known_person_multi(data: MultiImagePersonFields):
    """Register a person with one or more face images.

    Each image is independently validated, embedded, and stored.  If any
    image fails the others still succeed (no partial-rollback).
    """
    try:
        if not cctv_monitor:
            raise HTTPException(
                status_code=503, detail="CCTV Monitor not initialized")

        if not data.name.strip():
            raise HTTPException(status_code=400, detail="Name is required")

        if not data.imagePaths or len(data.imagePaths) == 0:
            raise HTTPException(
                status_code=400, detail="At least one image path is required")

        is_url = any(
            p.startswith(('http://', 'https://')) for p in data.imagePaths)

        logging.info(
            f"Multi-image registration: {data.name} "
            f"({len(data.imagePaths)} images, URL={is_url})")
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        result = reciveFromUi_multi(
            data.name,
            data.imagePaths,
            data.age,
            data.gender,
            data.role,
            data.socialnumber,
            is_url,
            device,
            cctv_monitor.face_handler,
            cctv_monitor.model,
            cctv_monitor.face_lock,
            cctv_monitor.model_lock,
            userwhom=data.userwhom,
            description=data.description,
            min_face_px=get_min_face_px(cctv_monitor),
        )

        # Refresh known names in CCTV monitor
        if cctv_monitor and result["success_count"] > 0:
            cctv_monitor.refresh_person(data.name)

        return {
            "success": result["success_count"] > 0,
            "message": (
                f"Registered {result['success_count']}/{result['total']} "
                f"face images for '{data.name}'"),
            "name": data.name,
            "details": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in multi-image registration: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/addFaceReference")
def add_face_reference(data: AddFaceReferenceFields):
    """Add an additional face reference image to an existing person.

    Detects the face, generates the embedding, and appends it to the
    person's embedding list (deduplicating near-identical embeddings).
    """
    try:
        if not cctv_monitor:
            raise HTTPException(
                status_code=503, detail="CCTV Monitor not initialized")

        if not data.name.strip():
            raise HTTPException(status_code=400, detail="Name is required")

        if not data.imagePath.strip():
            raise HTTPException(
                status_code=400, detail="Image path is required")

        is_url = data.imagePath.startswith(('http://', 'https://'))
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Resolve URL to local path if needed
        image_path = data.imagePath
        if is_url:
            local_path = urllib.request.urlretrieve(
                image_path, f"uploads/ref-{data.name}-{int(time.time())}.jpg")
            image_path = local_path[0]

        logging.info(
            f"Adding face reference for '{data.name}': {image_path}")

        min_face_px = get_min_face_px(cctv_monitor)
        embedding, meta = extract_face_embedding(
            image_path, cctv_monitor.face_handler,
            cctv_monitor.face_lock, cctv_monitor.model,
            cctv_monitor.model_lock, device, min_face_px,
            return_meta=True)

        if embedding is None:
            raise HTTPException(
                status_code=400,
                detail="No valid face detected in image (too small or none)")

        if not validate_face_embedding(embedding):
            raise HTTPException(
                status_code=400, detail="Invalid face embedding generated")

        # Fetch existing person data for age/gender/role/socialnumber
        from newsavatoDb import find_person_record
        record = find_person_record(data.name)
        if not record:
            raise HTTPException(
                status_code=404,
                detail=f"Person '{data.name}' not found. "
                       f"Create them first with /insertKToDp.")

        ok = add_embedding_to_person(
            data.name, embedding, image_path,
            record.get('age', ''), record.get('gender', ''),
            record.get('role', ''), record.get('socialnumber', ''),
            userwhom=record.get('userwhom', ''),
            description=record.get('description', ''),
            face_handler=_get_crop_face_handler(),
            face_lock=_crop_face_lock,
            model=cctv_monitor.model if cctv_monitor else None,
            model_lock=cctv_monitor.model_lock if cctv_monitor else None,
            meta=meta)

        if ok and cctv_monitor:
            cctv_monitor.refresh_person(data.name)

        return {
            "success": ok,
            "message": (
                f"Face reference added for '{data.name}'"
                if ok else "Failed to add face reference"),
            "name": data.name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error adding face reference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/removeFaceReference")
def remove_face_reference(data: RemoveFaceReferenceFields):
    """Remove a specific face reference image by its embedding index."""
    try:
        if not cctv_monitor:
            raise HTTPException(
                status_code=503, detail="CCTV Monitor not initialized")

        logging.info(
            f"Removing face reference for '{data.name}' "
            f"index={data.embeddingIndex}")

        ok = remove_person_embedding(data.name, data.embeddingIndex)

        if ok and cctv_monitor:
            cctv_monitor.refresh_person(data.name)

        return {
            "success": ok,
            "message": (
                f"Face reference removed for '{data.name}'"
                if ok else "Failed to remove face reference"),
            "name": data.name,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error removing face reference: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/d/known-persons")
async def get_known_persons():
    """Get list of known persons with embedding counts"""
    if not cctv_monitor:
        raise HTTPException(
            status_code=503, detail="CCTV Monitor not initialized")

    try:
        known_persons = []
        for name, data in cctv_monitor.known_names.items():
            known_persons.append({
                "name": name,
                "age": data.get('age', 'Unknown'),
                "gender": data.get('gender', 'Unknown'),
                "role": data.get('role', 'Unknown'),
                "socialnumber": data.get('socialnumber', ''),
                "embedding_count": len(data.get('embeddings', []))
            })

        return {
            "success": True,
            "count": len(known_persons),
            "persons": known_persons
        }

    except Exception as e:
        logging.error(f"Error getting known persons: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/known-persons/{person_name}/faces")
async def get_person_face_details(person_name: str):
    """Get detailed face information for a specific person."""
    try:
        info = get_person_faces(person_name)
        if info is None:
            raise HTTPException(
                status_code=404,
                detail=f"Person '{person_name}' not found")
        return {"success": True, "person": info}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error getting person faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/known-persons/{person_name}")
async def delete_known_person(person_name: str):
    """Delete a known person and all their face references."""
    try:
        if not cctv_monitor:
            raise HTTPException(
                status_code=503, detail="CCTV Monitor not initialized")

        from newsavatoDb import delete_person_from_db
        ok = delete_person_from_db(person_name)

        if ok and cctv_monitor:
            # Evict from the in-memory index immediately; refresh_person
            # would no-op here because the DB record is already gone.
            cctv_monitor.remove_person(person_name)

        return {
            "success": ok,
            "message": (
                f"Person '{person_name}' deleted"
                if ok else f"Failed to delete '{person_name}'")
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error deleting person: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/system/status")
def get_system_status():
    """Get system status information"""
    if not cctv_monitor:
        return {"status": "CCTV Monitor not initialized"}

    try:
        import psutil
        import torch

        # System info
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()

        status = {
            "system": {
                "cpu_usage": f"{cpu_percent}%",
                "memory_usage": f"{memory.percent}%",
                "memory_available": f"{memory.available / (1024**3):.2f} GB"
            },
            "cctv_monitor": {
                "device": cctv_monitor.device,
                "known_persons": len(cctv_monitor.known_names),
                "active_cameras": len(camera_registry),
            },
            "db_worker": get_db_worker().stats,
        }

        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(
                0).total_memory / (1024**3)
            gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
            status["gpu"] = {
                "available": True,
                "total_memory": f"{gpu_memory:.2f} GB",
                "used_memory": f"{gpu_memory_used:.2f} GB"
            }
        else:
            status["gpu"] = {"available": False}

        return status

    except ImportError:
        return {"status": "System monitoring not available (psutil not installed)"}
    except Exception as e:
        logging.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get('/util/refreshDb')
def refreshTheDb():
    """Refreshing the Known Face Db"""
    if cctv_monitor:
        cctv_monitor.known_names = cctv_monitor.load_db()
        cctv_monitor._build_embedding_index()
        logging.info("Known names refreshed in CCTV monitor,")
    return {"success": True}


@app.get("/util/imageSearch")
def querySearch(fileLocation: str):
    """Search for similar images"""
    if not cctv_monitor:
        raise HTTPException(status_code=503, detail="CCTV Monitor not initialized")

    logging.info(f"Image search query: {fileLocation}")
    # Incremental: only files added since the last index build get embedded
    cctv_monitor.precompute_embeddings()
    if not (os.path.exists(cctv_monitor.EMBEDDING_FILE)
            and os.path.exists(cctv_monitor.FILENAMES_FILE)):
        return []
    embeddings, filenames = cctv_monitor.load_embeddings()
    if len(filenames) == 0:
        return []
    query_path = fileLocation.replace('\\', '/')
    query_embedding = cctv_monitor.get_embedding(query_path)
    results = cctv_monitor.find_similar_images(
        query_embedding, embeddings, filenames, top_k=10)
    if results == []:
        return []
    try:
        response = requests.get(
            'http://127.0.0.1:8091/api/collections/collection/records', timeout=5)
        res = response.json()['items']
    except Exception as e:
        logging.error(f"Error fetching records for image search: {e}")
        return []

    by_filename = {item['filename']: item['id']
                   for item in res if item.get('filename')}
    ids = [by_filename[fname] for fname, _ in results if fname in by_filename]

    logging.info(ids)
    return ids

app.mount("/web/app", StaticFiles(directory="build/web",
          html=True), name="flutter")


def readPort():
    with open('hostname.json', 'r') as file:
        data = json.load(file)
        return data['port']


if __name__ == "__main__":

    host = '0.0.0.0'
    port = int(readPort())

    logging.info(f"Starting server on {host}:{port}")
    try:
        uvicorn.run(
            "app:app",
            host=host,
            port=port,
            log_level='info',
            log_config=None,
            reload=False,
            access_log=True
        )
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
    except Exception as e:
        logging.error(f"Server error: {e}")
        raise
