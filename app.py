
import asyncio
import json
import logging
import os
import shutil
import socket
import threading
import time
import webbrowser
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
# Import your improved CCtvMonitor class
from newengine import CCtvMonitor, image_crop,CameraManager
from onvifmaneger import get_rtsp_url
from savatoDb import reciveFromUi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

camera_registry = {}
camera_registry_lock = threading.Lock()
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


class RelayConfig(BaseModel):
    ip: str
    port: int
    username: str
    password: str
# Global CCTV monitor instance


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global cctv_monitor
    cctv_monitor = CCtvMonitor()

    # Startup
    logging.info("Starting CCTV Monitor application...")
    try:

        # Start the recognition worker
        # cctv_monitor.start()
        logging.info("CCTV Monitor initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize CCTV Monitor: {e}")
        raise

    yield

    # Shutdown
    logging.info("Shutting down CCTV Monitor application...")
    if cctv_monitor:
        await cctv_monitor.graceful_shutdown()
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
    return {
        "status": "healthy",
        "cctv_monitor_active": cctv_monitor is not None,
        "timestamp": time.time(),
        "multiprocessing": multiprocessing.cpu_count()
    }


@app.get("/{camera_id}")
async def video_feed(
    camera_id: str,
    request: Request,
    source: str = Query(...),
    role: bool = Query(False)
):
    if source == "0":
        source = int(source)
    camera_idx = int(camera_id[2:])
    # get or create camera
    with camera_registry_lock:
        if source not in camera_registry:
            camera_registry[source] = CameraManager(source, cctv_monitor,camera_idx,role)

        cam = camera_registry[source]

    cam.add_client()

    async def watch_disconnect():
        while True:
            if await request.is_disconnected():
                cam.remove_client()
                break
            await asyncio.sleep(0.2)

    asyncio.create_task(watch_disconnect())

    return StreamingResponse(
        cam.sendFrames(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )




def discover_onvif_stream():
    """Discover ONVIF cameras on the network"""
    ip_base = "192.168.1"

    def event_generator():
        # yield f"data: {json.dumps({'status': 'scanning', 'message': 'Starting network scan...'})}\n\n"

        found_devices = 0
        for i in range(1, 255):
            ip = f"{ip_base}.{i}"
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(0.3)
                result = sock.connect_ex((ip, 80))
                if result == 0:
                    found_devices += 1
                    yield f"data: {json.dumps({'ip': ip, 'port': 80, 'status': 'found'})}\n\n"
                sock.close()
            except Exception as e:
                logging.debug(f"Error scanning {ip}: {e}")
                continue

            # Send progress updates
            # if i % 50 == 0:
            #     progress = (i / 254) * 100
            #     yield f"data: {json.dumps({'status': 'progress', 'progress': progress, 'found': found_devices})}\n\n"

            time.sleep(0.1)

        # yield f"data: {json.dumps({'status': 'complete', 'total_found': found_devices})}\n\n"

    return event_generator()


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
async def get_camera_rtsp(request: RtspFields):
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


@app.post("/upload")
async def upload_file(isSearch: bool, file: UploadFile = File(...)):
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

    except Exception as e:
        # Clean up file if processing failed
        if os.path.exists(file_location):
            os.remove(file_location)
        logging.error(f"Error processing uploaded file: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/insertKToDp")
async def insert_known_person(data: KnownPersonFields):
    print("HEREEE")
    """Insert known person data to database"""
    try:
        # Validate required fields
        if not data.name.strip():
            raise HTTPException(status_code=400, detail="Name is required")

        if not data.imagePath.strip():
            raise HTTPException(
                status_code=400, detail="Image path is required")

        # Check if image path is URL or local path
        is_url = data.imagePath.startswith(('http://', 'https://'))

        logging.info(f"Inserting known person: {data.name} (URL: {is_url})")

        # Call database insertion function
        result = reciveFromUi(
            data.name,
            data.imagePath,
            data.age,
            data.gender,
            data.role,
            data.socialnumber,
            is_url
        )

        # Refresh known names in CCTV monitor
        if cctv_monitor:
            cctv_monitor.known_names = cctv_monitor.load_db()
            logging.info("Known names refreshed in CCTV monitor")

        return {
            "success": True,
            "message": "Person added successfully",
            "name": data.name
        }

    except Exception as e:
        logging.error(f"Error inserting known person: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/known-persons")
async def get_known_persons():
    """Get list of known persons"""
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


@app.delete("/known-persons/{person_name}")
async def delete_known_person(person_name: str):
    """Delete a known person (placeholder - implement in your database module)"""
    # This would need to be implemented in your database module
    raise HTTPException(
        status_code=501, detail="Delete functionality not implemented")


@app.get("/system/status")
async def get_system_status():
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
                "recognition_queue_size": cctv_monitor.recognition_queue.qsize()
            }
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


@app.get("/util/imageSearch")
async def querySearch(fileLocation: str):

    print(fileLocation)
    cctv_monitor.precompute_embeddings(
        cctv_monitor.load_image_searcher_model(), cctv_monitor.FOLDER_PATH)
    embeddings, filenames = cctv_monitor.load_embeddings()
    query_path = fileLocation.replace('\\', '/')
    query_embedding = cctv_monitor.get_embedding(query_path)
    results = cctv_monitor.find_similar_images(
        query_embedding, embeddings, filenames, top_k=10)
    if results == []:
        return []
    response = requests.get(
        'http://127.0.0.1:8091/api/collections/collection/records')
    res = response.json()['items']
    ids = []
    for fname, score in results:
        for json in res:
            if fname == json['filename']:
                ids.append(json['id'])

    logging.info(ids)
    return ids

app.mount("/web/app", StaticFiles(directory="build/web",
          html=True), name="flutter")


def updatePort():
    port = cctv_monitor.loadConfig()[3]
    with open('hostname.json', 'w') as file:
        json.dump({'port': port}, file, indent=4)
    return 0


def readPort():
    with open('hostname.json', 'r') as file:
        data = json.load(file)
        return data['port']


if __name__ == "__main__":

    # cctv_monitor =CCtvMonitor()
    host = '0.0.0.0'
    # port =int(cctv_monitor.loadConfig()[3])
    port = int(readPort())
    # TODO:ADD HOSTNAME AND PORT AND SEE IF ANY THING CHANGE IN THAT CODE

    logging.info(f"Starting server on {host}:{port}")
    # webbrowser.open(f'http://127.0.0.1:{port}/web/app')
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
