
import json
import logging
import os
import shutil
import socket
import threading
import time
import webbrowser
import base64
import asyncio
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, File, Query, Request, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.middleware.gzip import GZipMiddleware
import requests
import uvicorn
import multiprocessing
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
# Import your improved CCtvMonitor class
from test2engine import CCtvMonitor, image_crop
from onvifmaneger import get_rtsp_url
from savatoDb import reciveFromUi

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s'
)

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

# Global CCTV monitor instance
cctv_monitor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global cctv_monitor
    
    # Startup
    logging.info("Starting CCTV Monitor application...")
    try:
        cctv_monitor = CCtvMonitor()
        # Start the recognition worker
        cctv_monitor.start()
        logging.info("CCTV Monitor initialized successfully")
    except Exception as e:
        logging.error(f"Failed to initialize CCTV Monitor: {e}")
        raise
    
    yield
    
    # Shutdown
    logging.info("Shutting down CCTV Monitor application...")
    if cctv_monitor:
        cctv_monitor.graceful_shutdown()
    logging.info("Application shutdown complete")

# Initialize thread pool for CPU-bound tasks
thread_pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() * 2)

# Cache for storing frequently accessed data
cache: Dict[str, dict] = {}

# Create FastAPI app with lifespan manager
app = FastAPI(
    lifespan=lifespan,
    title="Face Recognition API",
    description="API for face detection and recognition",
    version="1.0.0"
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS configuration
origins = ["*"]  # Change this to specific domains in production

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Origin", "X-Requested-With", "Content-Type", "Accept"],
)

# Custom middleware for request timing and logging
class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response

app.add_middleware(TimingMiddleware)

@app.get("/health")
@lru_cache(maxsize=1)  # Cache for 60 seconds
async def health_check():
    """Health check endpoint with system metrics"""
    import psutil
    
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "status": "healthy",
            "cctv_monitor_active": cctv_monitor is not None,
            "timestamp": time.time(),
            "system": {
                "cpu_usage": f"{cpu_percent}%",
                "memory_usage": f"{memory.percent}%",
                "memory_available": f"{memory.available / (1024**3):.2f} GB",
                "disk_usage": f"{disk.percent}%",
                "cpu_cores": multiprocessing.cpu_count()
            }
        }
    except Exception as e:
        logging.error(f"Health check error: {e}")
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": time.time()
        }

@app.get("/{camera_id}")
async def video_feed(camera_id: str, request: Request, source: str = Query(...), role :bool = Query()):
    """Stream video from a specific camera"""
    if not cctv_monitor:
        raise HTTPException(status_code=503, detail="CCTV Monitor not initialized")
    
    if not camera_id.startswith("rt"):
        raise HTTPException(
            status_code=400, 
            detail="Invalid camera ID format. Use rt1, rt2, etc."
        )

    try:
        # Handle local camera (source='0' becomes integer 0)
        if source == '0':
            source = int(source)
        
        # Extract camera index from ID (rt1 -> 1)
        camera_idx = int(camera_id[2:])
        
        logging.info(f"Starting video stream for camera {camera_idx} with source: {source}")
        if not role:
            
            if threading.Thread(
                target=cctv_monitor.recognition_worker,
                daemon=True
            ).is_alive():
                pass
            else:
                threading.Thread(
                target=cctv_monitor.recognition_worker,
                daemon=True
            ).start()
            
                
        

        return StreamingResponse(
            cctv_monitor.generate_frames(camera_idx, source, request,role),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store",
                "Connection": "close"
            }
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid camera ID: {camera_id}. Use format: rt1, rt2, etc. Error: {str(e)}"
        )
    except Exception as e:
        logging.error(f"Error in video_feed: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

def discover_onvif_stream():
    """Discover ONVIF cameras on the network"""
    ip_base = "192.168.1"

    def event_generator():
        yield f"data: {json.dumps({'status': 'scanning', 'message': 'Starting network scan...'})}\n\n"
        
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
            if i % 50 == 0:
                progress = (i / 254) * 100
                yield f"data: {json.dumps({'status': 'progress', 'progress': progress, 'found': found_devices})}\n\n"
            
            time.sleep(0.1)
        
        yield f"data: {json.dumps({'status': 'complete', 'total_found': found_devices})}\n\n"
    
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
        logging.info(f"Getting RTSP URL for camera at {request.ip}:{request.port}")
        
        port = int(request.port)
        rtsp_url = get_rtsp_url(request.ip, port, request.username, request.password)
        
        if not rtsp_url:
            raise HTTPException(status_code=404, detail="Could not retrieve RTSP URL")
        
        return {'rtsp': rtsp_url}
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid port number")
    except Exception as e:
        logging.error(f"Error getting RTSP URL: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get RTSP URL: {str(e)}")

async def process_upload(file_location: str, is_search: bool) -> dict:
    """Process uploaded file in background"""
    try:
        img_encoded = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: image_crop(file_location, is_search)
        )
        
        if img_encoded is None:
            os.remove(file_location)
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')
        return {
            "success": True,
            "file_location": file_location,
            "filename": os.path.basename(file_location),
            "image_data": img_base64,
            "media_type": "image/jpeg"
        }
    except Exception as e:
        if os.path.exists(file_location):
            os.remove(file_location)
        raise e

@app.post("/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    isSearch: bool = Query(..., description="Whether this is a search request"),
    file: UploadFile = File(...)
):
    """Upload and process image file with background processing"""
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
    
    # Generate unique filename with better uniqueness
    timestamp = int(time.time() * 1000)  # millisecond precision
    filename = f"{timestamp}_{os.urandom(4).hex()}_{file.filename}"
    file_location = os.path.join(UPLOAD_DIR_VIDEO, filename)

    try:
        # Save uploaded file
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"File uploaded: {file_location}")
        
        # Process image to crop face

            
        img_encoded = image_crop(file_location,isSearch)
        
        
        
        
        if img_encoded is None:
            # Clean up file if processing failed
            os.remove(file_location)
            raise HTTPException(status_code=400, detail="No face detected in image")
        
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
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.post("/insertKToDp")
async def insert_known_person(data: KnownPersonFields):
    print("HEREEE")
    """Insert known person data to database"""
    try:
        # Validate required fields
        if not data.name.strip():
            raise HTTPException(status_code=400, detail="Name is required")
        
        if not data.imagePath.strip():
            raise HTTPException(status_code=400, detail="Image path is required")
        
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
        raise HTTPException(status_code=503, detail="CCTV Monitor not initialized")
    
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
    raise HTTPException(status_code=501, detail="Delete functionality not implemented")




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
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
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

# Cache for embeddings with 10-minute expiry
@lru_cache(maxsize=1)
async def get_cached_embeddings():
    """Get cached embeddings and filenames"""
    return await asyncio.get_event_loop().run_in_executor(
        thread_pool,
        lambda: cctv_monitor.load_embeddings()
    )

async def fetch_collection_records():
    """Fetch collection records with retry logic"""
    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            async with requests.Session() as session:
                response = await session.get('http://127.0.0.1:8091/api/collections/collection/records')
                response.raise_for_status()
                return response.json()['items']
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(retry_delay * (attempt + 1))

@app.get("/util/imageSearch")
async def querySearch(fileLocation: str, top_k: int = Query(default=10, le=50)):
    """Search for similar images with optimized performance"""
    try:
        # Normalize file path
        query_path = fileLocation.replace('\\', '/')
        
        # Get embeddings (cached)
        embeddings, filenames = await get_cached_embeddings()
        
        # Get query embedding in thread pool
        query_embedding = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: cctv_monitor.get_embedding(query_path)
        )
        
        # Find similar images
        results = await asyncio.get_event_loop().run_in_executor(
            thread_pool,
            lambda: cctv_monitor.find_similar_images(query_embedding, embeddings, filenames, top_k=top_k)
        )
        
        # Fetch collection records
        records = await fetch_collection_records()
        
        # Create filename to ID mapping for O(1) lookup
        filename_to_id = {record['filename']: record['id'] for record in records}
        
        # Get IDs efficiently
        ids = [filename_to_id[fname] for fname, _ in results if fname in filename_to_id]
        
        logging.info(f"Found {len(ids)} matching images")
        return ids
        
    except Exception as e:
        logging.error(f"Image search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

async def cleanup():
    """Cleanup function for graceful shutdown"""
    logging.info("Cleaning up resources...")
    thread_pool.shutdown(wait=True)
    if cctv_monitor:
        await asyncio.get_event_loop().run_in_executor(None, cctv_monitor.graceful_shutdown)

if __name__ == "__main__":
    host = '0.0.0.0'
    port = 8000
    workers = min(multiprocessing.cpu_count() * 2 + 1, 8)  # Optimal worker count
    
    logging.info(f"Starting server on {host}:{port} with {workers} workers")
    
    config = uvicorn.Config(
        "test2api:app",
        host=host,
        port=port,
        workers=workers,
        log_level="info",
        log_config=None,
        reload=False,
        access_log=True,
        limit_concurrency=1000,  # Limit concurrent connections
        timeout_keep_alive=30,   # Keep-alive timeout
        ssl_keyfile=None,        # Add SSL config if needed
        ssl_certfile=None
    )
    
    try:
        server = uvicorn.Server(config)
        server.run()
    except KeyboardInterrupt:
        logging.info("Server stopped by user")
        asyncio.run(cleanup())
    except Exception as e:
        logging.error(f"Server error: {e}")
        asyncio.run(cleanup())
        raise